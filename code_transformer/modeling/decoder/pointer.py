from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence

from code_transformer.configuration.attention import AttentionType


class PointerNetwork(nn.Module):

    def __init__(self, d_model, subtokens_per_token, pointer_attention_type: AttentionType, n_attention_heads=8):
        super(PointerNetwork, self).__init__()

        self.d_model = d_model
        self.pointer_attention_type = pointer_attention_type

        # "Embedding" of the sentinel used for computing the logit of the gate
        self.sentinel = nn.Parameter(torch.Tensor(self.d_model, 1))

        # Linear transformation for computing the query from the LSTM hidden state
        self.query_linear = nn.Linear(self.d_model, self.d_model)

        # Linear transformation for getting n subtokens out of the representations of the final encoder layer
        self.subtoken_extractor_linear = nn.Linear(self.d_model, subtokens_per_token * self.d_model)

        if self.pointer_attention_type == AttentionType.ADDITIVE:
            self.additive_attention_W = nn.Linear(self.d_model * 2, self.d_model)  # bidirectional
            self.additive_attention_tanh = nn.Tanh()
            self.additive_attention_v = nn.Parameter(torch.Tensor(self.d_model, 1))  # context vector
        elif self.pointer_attention_type == AttentionType.MULTIHEAD:
            self.multihead_attention = MultiheadAttention(self.d_model, n_attention_heads)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    #         pointer_input_subtokens = transformer_output.all_emb[-1][0].transpose(0, 1)
    def init_batch(self, pointer_input_subtokens, pointer_pad_mask, extended_vocabulary_ids, len_vocab):
        bsz = pointer_input_subtokens.shape[0]
        seq_len = pointer_input_subtokens.shape[1]
        device = next(self.parameters()).device

        # Pointer Sentinel
        # Take embeddings from last encoder layer, transform them to n_subtokens x d_model and then slice out
        # Each subtoken. This is necessary as the pointer query needs to have one representation per subtoken for
        # dot-product computation. PAD subtokens are thrown away which results in different length subtoken
        # sequences. These are then padded in the end with [0, ..., 0] representation vectors
        # all_emb[-1][0] takes the content stream of the last layer, all_emb[-1][1] would take query stream

        pointer_input_subtokens = self.subtoken_extractor_linear(pointer_input_subtokens).view(bsz, seq_len,
                                                                                               self.d_model, -1)
        pointer_input_subtokens = pointer_input_subtokens.transpose(2, 3)
        pointer_input_subtokens = [s[m] for s, m in zip(pointer_input_subtokens, pointer_pad_mask)]
        # bsz x seq_len_subtokens
        pointer_subtoken_mask = pad_sequence(
            [torch.ones((s.shape[0],), dtype=torch.bool, device=device) for s in pointer_input_subtokens],
            batch_first=True, padding_value=False)
        # Append ones to subtoken mask to always pass sentinel
        self.pointer_subtoken_mask = torch.cat(
            [pointer_subtoken_mask, torch.ones((bsz, 1), dtype=torch.bool, device=device)], dim=1)
        # bsz x seq_len_subtokens x d_model
        pointer_input_subtokens = pad_sequence(pointer_input_subtokens, batch_first=True)
        self.seq_len_subtokens = pointer_input_subtokens.shape[1]

        # seq_len x d_model => seq_len x d_model x n_subtokens => seq_len_subtokens x d_model
        # Assume input tokens don't have unknown tokens (extended vocabulary)
        # Add the sentinel representation to the end of each subtoken sequence
        self.len_vocab = len_vocab
        self.len_extended_vocab = max(len_vocab, extended_vocabulary_ids.max().item()) + 1
        self.pointer_input_embeddings = torch.cat(
            [pointer_input_subtokens, self.sentinel.unsqueeze(0).expand(bsz, self.d_model, 1).transpose(1, 2)], dim=1)
        self.extended_vocabulary_ids = extended_vocabulary_ids

    def calculate_pointer_attention(self, pointer_query: torch.Tensor):
        bsz = pointer_query.shape[0]
        device = next(self.parameters()).device

        # Use last decoder hidden state to produce query vector for pointer
        # bsz x d_model
        pointer_query = self.query_linear(pointer_query)
        pointer_query = torch.tanh(pointer_query)
        pointer_query = pointer_query.unsqueeze(-1)

        # Pointer attention between query and input embeddings
        pointer_attention = None
        if self.pointer_attention_type == AttentionType.SCALED_DOT_PRODUCT:
            # Regular dot-product attention
            pointer_attention = torch.bmm(self.pointer_input_embeddings, pointer_query)

            # Scaled dot-product attention
            pointer_attention = pointer_attention / sqrt(self.d_model)

            pointer_attention = pointer_attention.squeeze(-1)
            pointer_attention[~self.pointer_subtoken_mask] = torch.finfo(torch.float).min
            pointer_attention = F.log_softmax(pointer_attention, dim=1)
        elif self.pointer_attention_type == AttentionType.ADDITIVE:
            # Additive Attention
            pointer_query = pointer_query.transpose(1, 2).repeat(1, self.seq_len_subtokens + 1, 1)
            pointer_attention = torch.cat([pointer_query, self.pointer_input_embeddings], dim=2)
            pointer_attention = self.additive_attention_tanh(self.additive_attention_W(pointer_attention))
            v = self.additive_attention_v.repeat(bsz, 1, 1)  # [H,1] -> [B,H,1]
            pointer_attention = torch.bmm(pointer_attention, v)  # [B,T,H]*[B,H,1] -> [B,T,1]

            pointer_attention = pointer_attention.squeeze(-1)
            pointer_attention[~self.pointer_subtoken_mask] = torch.finfo(torch.float).min
            pointer_attention = F.log_softmax(pointer_attention, dim=1)
        elif self.pointer_attention_type == AttentionType.MULTIHEAD:
            # Multihead attention
            pointer_attention = self.multihead_attention(pointer_query.permute(2, 0, 1),
                                                         self.pointer_input_embeddings.transpose(0, 1),
                                                         self.pointer_input_embeddings.transpose(0, 1),
                                                         key_padding_mask=~self.pointer_subtoken_mask)

            pointer_attention = pointer_attention[1]
            pointer_attention = pointer_attention.squeeze(1)
            # pointer_attention = F.log_softmax(pointer_attention, dim=1)
            pointer_attention[~self.pointer_subtoken_mask] = torch.finfo(torch.float).eps
            pointer_attention = pointer_attention.log()

        if torch.isnan(pointer_attention).any() or (pointer_attention == -float('inf')).any():
            print('NaN in pointer attention after softmax!', pointer_attention)
        pointer_gate = pointer_attention[:, -1].unsqueeze(-1)

        self.pointer_attention_distribution = pointer_attention[:, :-1]

        # Sum up logits of the same tokens
        # bsz x len_extended_vocab
        M = torch.zeros((bsz, self.len_extended_vocab, self.seq_len_subtokens), device=device)
        M[
            torch.arange(bsz).unsqueeze(-1).expand(bsz, self.seq_len_subtokens).reshape(-1),
            self.extended_vocabulary_ids.view(-1),
            torch.arange(self.seq_len_subtokens).repeat(bsz)] = 1
        pointer_attention = torch.bmm(M, pointer_attention[:, :-1].unsqueeze(-1).exp()).squeeze()
        pointer_attention = (pointer_attention + torch.finfo(torch.float).eps).log()
        pointer_attention = pointer_attention - torch.log1p(
            -pointer_gate.exp().view(bsz, -1) + torch.finfo(torch.float).eps)

        # Avoid having -inf in attention scores as they produce NaNs during backward pass
        pointer_attention[pointer_attention == -float('inf')] = torch.finfo(torch.float).min

        if torch.isnan(pointer_attention).any():
            print("NaN in final pointer attention!", pointer_attention)

        self.pointer_gate = pointer_gate
        self.pointer_attention = pointer_attention

    def combine_probabilites(self, subtoken_logits: torch.Tensor):
        # Probability distribution of the decoder over original vocabulary
        # bsz x vocab_size
        log_probs = F.log_softmax(subtoken_logits.view(-1, subtoken_logits.size(-1)), dim=1)

        # Decoder cannot predict extended vocabulary tokens. Thus, these have 0 probability
        log_probs = F.pad(log_probs, [0, self.len_extended_vocab - self.len_vocab], value=-float('inf'))

        # Combine decoder probability distribution with pointer attention distribution in log space
        p = torch.stack([log_probs + self.pointer_gate,
                         self.pointer_attention + (1 - self.pointer_gate.exp() + torch.finfo(torch.float).eps).log()])
        log_probs = torch.logsumexp(p, dim=0)

        subtoken_logits = log_probs

        return subtoken_logits

    def get_next_input(self, next_input: torch.Tensor, unk_id):
        device = next(self.parameters()).device

        # If an extended vocabulary word was chosen, it has to be replaced with <unk> as extended
        # vocabulary words do not have an embedding
        return torch.where(next_input < self.len_vocab, next_input, torch.tensor([unk_id], device=device))
