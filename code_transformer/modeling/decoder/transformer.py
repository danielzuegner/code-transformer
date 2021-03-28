import math
from typing import Optional

import torch
from torch import nn
from torch.nn import TransformerDecoder, MultiheadAttention, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.modules.transformer import TransformerDecoderLayer

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.code_transformer.code_transformer import _get_activation_fn
from code_transformer.modeling.code_transformer.lm import CodeTransformerOutput
from code_transformer.modeling.decoder.pointer import PointerNetwork
from code_transformer.utils.data import batch_index_select


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, position):
        x = x + self.pe[position, :]
        return self.dropout(x)


class TransformerLMDecoder(nn.Module):

    def __init__(self, config: TransformerLMDecoderConfig):
        super(TransformerLMDecoder, self).__init__()

        self.lm_encoder = config.lm_encoder

        self.sos_id = config.sos_id
        self.unk_id = config.unk_id
        self.output_subtokens_per_token = config.output_subtokens_per_token
        self.use_separate_vocab = config.target_vocab_size is not None
        # If target_vocab_size is set, a separate vocabulary for input and output tokens is assumed
        self.vocab_size = config.target_vocab_size if self.use_separate_vocab else config.lm_encoder.vocab_size
        self.d_model = config.lm_encoder.d_model
        n_heads = config.decoder_nhead

        self.n_layers = config.n_layers
        self.use_teacher_forcing = config.use_teacher_forcing

        self.output_nonlinearity = None
        if config.output_nonlinearity is not None:
            self.output_nonlinearity = _get_activation_fn(config.output_nonlinearity)

        self.loss_fct = config.loss_fct

        self.use_pointer_network = config.use_pointer_network
        self.use_pointer_query_linear = config.use_pointer_query_linear
        self.use_pointer_query_self_attention = config.use_pointer_query_self_attention
        self.concat_query_and_pointer = config.concat_query_and_pointer
        self.attend_cls_token = config.attend_cls_token

        decoder_layer = TransformerDecoderLayer(self.d_model, config.decoder_nhead, config.decoder_dim_feedforward,
                                                config.decoder_dropout, config.decoder_activation)
        self.transformer_decoder = TransformerDecoder(decoder_layer, self.n_layers)

        self.positional_encoding = PositionalEncoding(self.d_model, config.decoder_dropout)

        if self.use_pointer_network:
            self.pointer_network = PointerNetwork(self.d_model, self.lm_encoder.subtokens_per_token,
                                                  config.pointer_attention_type,
                                                  n_heads)

            if self.concat_query_and_pointer:
                self.pointer_query_linear = nn.Linear(self.d_model * 2, self.d_model)
                self.pointer_query_nonlinearity = _get_activation_fn('tanh')

            if self.use_pointer_query_self_attention:
                self.pointer_query_self_attention = MultiheadAttention(self.d_model, n_heads,
                                                                       dropout=config.decoder_dropout)
                self.pointer_query_norm = LayerNorm(self.d_model)

        if self.use_separate_vocab:
            self.target_token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self,
                labels=None,
                extended_vocabulary_ids=None,
                pointer_pad_mask: Optional[torch.Tensor] = None,
                **model_input) -> CodeTransformerOutput:
        """
        :param labels:
        :param extended_vocabulary_ids: torch.Tensor [B, subtoken_seq_len]
            Defines a sequence of subtokens for every sample. Can be seen as a flattened version of the tokens input
            with UNKNOWN_TOKENs replaced by incremental artificial vocabulary IDs that are only valid for this sample.
            Needed for the pointer mechanism
        :param pointer_pad_mask: torch.Tensor [B, S, num_subtokens]
            A mask that specifies padding down to subtoken level. Needed for the pointer mechanism as we need to point
            to distinct subtokens. 1 indicates that the respective subtoken is NOT a PAD token
        :param model_input:
            additional inputs that are passed on to the encoder.
            The TransformerDecoder expects that model_input has at least the following entries:
                - pad_mask: [B, S], where 1 indicates that the position is a PAD token
                - attention_mask: [B, S, S], where a 1 at [:,i,j] indicates that position i may not attend position j
        :return:
        """

        device = next(self.parameters()).device
        pointer_gates = None
        pointer_attentions = None
        pointer_attention_distributions = None

        transformer_output = self.lm_encoder.forward(need_all_embeddings=True, **model_input)

        B = transformer_output[0].shape[0]
        S = transformer_output.all_emb[-1][0].shape[0]
        V = self.vocab_size
        D = self.d_model

        # all_emb[-1][0] takes the content stream of the last layer, all_emb[-1][1] would take query stream
        content_stream_emb = transformer_output.all_emb[-1][0].transpose(0, 1)  # [B, S, D]
        query_stream_emb = transformer_output.all_emb[-1][1]  # [n_predict, B, D]
        n_predict = transformer_output[0].shape[1]
        if n_predict > 1:
            content_stream_emb = content_stream_emb \
                .unsqueeze(1) \
                .repeat((1, n_predict, 1, 1)) \
                .reshape(B * n_predict, S, D)
            labels = labels.reshape(B * n_predict, 1, -1)
            B = B * n_predict

        # Initially start decoding with a sequence containing only one <s> token per sample
        # Input tokens should have B x T x D
        # every batch decoding starts with the same initial input
        initial_input = torch.tensor([[self.sos_id]], device=device)
        token_embedding = self.target_token_embedding if self.use_separate_vocab else self.lm_encoder.token_embedding
        decoder_input = token_embedding(initial_input).expand((B, -1, -1))
        decoder_input = self.positional_encoding.forward(decoder_input, 0)

        if self.use_pointer_network:
            pointer_input_subtokens = content_stream_emb
            if n_predict > 1:
                pointer_pad_mask = pointer_pad_mask.unsqueeze(1) \
                    .repeat(1, n_predict, 1, 1) \
                    .reshape(B, S, -1)
                extended_vocabulary_ids = extended_vocabulary_ids.unsqueeze(1) \
                    .repeat(1, n_predict, 1) \
                    .reshape(B, -1)

            self.pointer_network.init_batch(pointer_input_subtokens, pointer_pad_mask, extended_vocabulary_ids,
                                            self.vocab_size)

            logits = torch.zeros((self.output_subtokens_per_token, B, self.pointer_network.len_extended_vocab),
                                 device=device)
            pointer_gates = torch.zeros((self.output_subtokens_per_token, B))
            pointer_attentions = torch.zeros(
                (self.output_subtokens_per_token, B, self.pointer_network.len_extended_vocab))
            pointer_attention_distributions = torch.zeros(
                (self.output_subtokens_per_token, B, extended_vocabulary_ids.shape[1])
            )
        else:
            logits = torch.zeros((self.output_subtokens_per_token, B, V), device=device)

        # pad_mask has 1s for all regular (non-pad) tokens
        # attention_mask has 1s for all illegal tokens that may not be attended (such as function name and CLS token)
        pad_mask = model_input['pad_mask'].bool()
        if n_predict > 1:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, n_predict, 1).reshape(B, -1)
            attention_mask = model_input['attention_mask']
            label_idx = torch.stack([torch.where(tm == 1)[0] for tm in model_input['target_mapping'].sum(dim=1)])
            attention_mask = batch_index_select(attention_mask, dim=1, index=label_idx)
            attention_mask = attention_mask.reshape(B, -1)
        else:
            attention_mask = model_input['attention_mask']
            attention_mask = torch.stack(
                [attention_mask[i][torch.where(model_input['pad_mask'][i] == 0)[0]].sum(dim=0) for i in range(B)])
        attention_mask = attention_mask > 0
        if self.attend_cls_token:
            attention_mask[:, 0] = False  # CLS token may be attended

        for idx in range(self.output_subtokens_per_token):

            if self.use_pointer_network:
                if self.concat_query_and_pointer:
                    pointer_query = decoder_input.select(1, -1)
                    pointer_query = torch.cat([pointer_query, query_stream_emb.reshape(B, D)], dim=1)

                    pointer_query = self.pointer_query_linear(pointer_query)
                    pointer_query = self.pointer_query_nonlinearity(pointer_query)
                else:
                    pointer_query = decoder_input.select(1, -1)

                if self.use_pointer_query_self_attention:
                    pointer_query = \
                        self.pointer_query_self_attention(pointer_query.unsqueeze(0), decoder_input.transpose(0, 1),
                                                          decoder_input.transpose(0, 1))[0]
                    pointer_query = self.pointer_query_norm(pointer_query)

                    pointer_query = pointer_query.squeeze(0)

                self.pointer_network.calculate_pointer_attention(pointer_query)

            decoder_output = self.transformer_decoder.forward(decoder_input.transpose(0, 1),
                                                              content_stream_emb.transpose(0, 1),
                                                              memory_key_padding_mask=pad_mask | attention_mask)
            if self.output_nonlinearity is not None:
                decoder_output = self.output_nonlinearity(decoder_output)

            # B x V
            subtoken_logits = decoder_output.select(0, -1) @ token_embedding.weight.T

            if self.use_pointer_network:
                subtoken_logits = self.pointer_network.combine_probabilites(subtoken_logits)
                pointer_gates[idx] = self.pointer_network.pointer_gate.squeeze(-1)
                pointer_attentions[idx] = self.pointer_network.pointer_attention
                pointer_attention_distributions[idx] = self.pointer_network.pointer_attention_distribution

            logits[idx] = subtoken_logits

            # Calculate next decoder_input
            if self.use_teacher_forcing and self.training:
                # Use previous label as next input
                next_input = labels[:, :, idx]  # B x 1
            else:
                next_input = subtoken_logits.argmax(-1).detach().unsqueeze(1)  # B x 1

            if self.use_pointer_network:
                next_input = self.pointer_network.get_next_input(next_input, self.unk_id)

            next_input_embedding = token_embedding(next_input)
            next_input_embedding = self.positional_encoding.forward(next_input_embedding, idx + 1)
            next_input = torch.cat([decoder_input, next_input_embedding], 1)
            decoder_input = next_input

        loss = self.loss_fct(logits.transpose(0, 1).reshape(-1, logits.size(-1)), labels.view(-1))

        logits = logits.transpose(0, 1).unsqueeze(1)  # B x 1 x output_subtokens x V
        logits = logits.reshape(B // n_predict, n_predict, logits.shape[2], logits.shape[3])
        outputs = CodeTransformerOutput(loss=loss,
                                        logits=logits,
                                        attentions=transformer_output.attentions,
                                        pointer_gates=pointer_gates,
                                        pointer_attentions=pointer_attentions,
                                        pointer_attention_distributions=pointer_attention_distributions)

        return outputs
