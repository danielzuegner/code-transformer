import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_
from code_transformer.configuration.great_transformer import GreatEncoderConfig
from typing import Optional
from code_transformer.preprocessing.dataset.code_summarization import GreatBatch


class AttentionLayer(nn.Module):
    """
        Implementation of multi-headed attention with optional edge-bias.

        This class supports self-attention and key-value attention, with (non-optional) masks. If bias_dim is not None,
         the attention computation(s) assumes that a (sparse) bias vector is provided, formatted like:
         (edge_type, batch_index, key_index, query_index). Bias edge types are embedded in the same dimension as each
          head's attention and projected to a scalar before being inserted into the attention computation as
          (q + b) * k.
    """

    def __init__(self, attention_dim, num_heads=None, hidden_dim=None, bias_dim=None):
        super(AttentionLayer, self).__init__()
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.attention_dim
        self.num_heads = 1 if num_heads is None else num_heads
        self.attention_dim_per_head = self.attention_dim // self.num_heads
        self.bias_dim = bias_dim

        self.attn_query = nn.Parameter(torch.empty([self.hidden_dim, self.num_heads, self.attention_dim_per_head]))
        self.attn_keys = nn.Parameter(torch.empty([self.hidden_dim, self.num_heads, self.attention_dim_per_head]))
        self.attn_values = nn.Parameter(torch.empty([self.hidden_dim, self.num_heads, self.attention_dim_per_head]))
        self.weight_out = nn.Parameter(torch.empty([self.num_heads, self.attention_dim_per_head, self.hidden_dim]))

        if self.bias_dim is not None:
            self.bias_embs = nn.Embedding(self.bias_dim, self.attention_dim_per_head)
            self.bias_scalar = nn.Parameter(torch.empty([self.attention_dim_per_head, 1]))

        self._reset_parameters()

    def forward(self, states, key_states, masks, attention_bias):
        # Compute key, query and value vectors, reshaped to [Batch, Heads, Time, Dim] where Dim is attention_dim//num_heads.
        query, keys, values = self.compute_qkv(states, key_states)

        # Compute attention weights, and context from these.
        alpha = self.get_attention_weights(query, keys, masks, attention_bias)

        # Compute weigthed context and project out.
        context = torch.einsum('bhqk,bkha->bqha', alpha, values)
        context = torch.einsum('btha,had->btd', context, self.weight_out)
        return context

    # Compute key, query and value vectors.
    def compute_qkv(self, states, key_states):
        query = torch.einsum('btd,dha->btha', states, self.attn_query)  # Queries are always computed on states
        keys = torch.einsum('btd,dha->btha', states if key_states is None else key_states, self.attn_keys)
        values = torch.einsum('btd,dha->btha', states if key_states is None else key_states, self.attn_values)
        return query, keys, values

    # Compute attention weights from cross-product between keys and queries (scaled, masked, softmaxed).
    def get_attention_weights(self, query, keys, masks, attention_bias):
        alpha = torch.einsum('bkha,bqha->bhqk', keys, query)

        # If bias_dim is set, assume that a bias vector is provided.
        if self.bias_dim is not None:
            # Embed edge types in per-head-attention dimension. Experimentally, mat-mul tends to be faster here,
            # but regular embedding is equally valid.
            bias = self.bias_embs(attention_bias[:, 0])
            # Project down to a scalar.
            bias = torch.matmul(bias, self.bias_scalar).squeeze(-1)
            alpha_shape = alpha.shape
            bias_shape = torch.tensor([alpha_shape[0], alpha_shape[2], alpha_shape[3]])
            # Scatter edge biases to their [batch_index, key_index, query_index] positions.
            bs = torch.zeros([alpha_shape[0], alpha_shape[2], alpha_shape[3]],
                             device=next(self.parameters()).device)
            bs[tuple(attention_bias[:, 1:].t())] = bias
            bias = bs
            # bias = torch.scatter_nd(attention_bias[:, 1:], bias, bias_shape)

            # Since bias is a scalar, we can reduce memory cost by rewriting the attention from (q + b) * k to q*k +
            # b*reduce_sum(k, -1)
            keys = keys.sum(-1)  # bkh
            bias = torch.einsum('bqk,bkh->bhqk', bias, keys)
            # Accordingly, simply add the bias as a residual to standard dot-product attention.
            alpha = alpha + bias

        # Scale and apply mask
        alpha *= (1 / torch.tensor(self.attention_dim_per_head,
                                   device=next(self.parameters()).device).to(torch.float32).sqrt())
        if masks is not None:
            masks = masks.unsqueeze(1)
            alpha = alpha - masks * torch.finfo(torch.float32).max
        alpha = F.softmax(alpha, dim=-1)
        return alpha

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            xavier_uniform_(p)


def positional_encoding(dim, sentence_length, dtype=torch.float32):
    posr = np.arange(sentence_length)
    dimr = np.arange(dim)
    encoded_vec = posr[:, None] / np.power(10000, 2 * dimr[None, :] / dim)
    encoded_vec = encoded_vec.flatten()
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return torch.tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class LayerNormalization(nn.Module):
    def __init__(self, hidden_dim):
        super(LayerNormalization, self).__init__()
        self.hidden_dim = hidden_dim
        self.scale = nn.Parameter(torch.ones(self.hidden_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(self.hidden_dim), requires_grad=True)
        self.build = True

    def forward(self, x: torch.Tensor, epsilon=1e-3):
        mean, variance = x.mean(-1, keepdim=True), x.var(-1, keepdim=True)
        norm_x = (x - mean) * (variance + epsilon).pow(-0.5)
        return norm_x * self.scale + self.bias


class Transformer(nn.Module):
    """Transformer language model: converts indices into hidden states through layers of multi-headed attention and
    feed-forward dense layers.

        Augments a generic Transformer with attentional bias, if bias_dim is provided. See documentation on
        AttentionLayer for more details.
        To generate language from the resulting states, pass the states to the "predict" function. Note that it
        assumes that the input vocabulary is output vocabulary (i.e., it reuses the model's embedding table).
    """
    NOOP_BIAS = torch.zeros((0, 4), dtype=torch.int32)

    def __init__(self, model_config, bias_dim=None):
        super(Transformer, self).__init__()
        self.bias_dim = model_config["bias_dim"]
        self.embed_dim = model_config["embed_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.is_encoder_decoder = model_config["is_encoder_decoder"]
        assert self.embed_dim == self.hidden_dim, "Embedding and hidden dimension must be equal for Transformer."
        self.ff_dim = model_config["ff_dim"]
        self.attention_dim = model_config["attention_dim"]
        self.num_layers = model_config["num_layers"]
        self.num_heads = model_config["num_heads"]
        self.dropout_rate = model_config["dropout_rate"]
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Initialize default positional encoding for very long sequences. Can make this a parameter if necessary.
        self.pos_enc = positional_encoding(model_config["embed_dim"], 5000)

        make_att = lambda: AttentionLayer(self.attention_dim, self.num_heads, self.hidden_dim, self.bias_dim)
        self.attention = nn.ModuleList([make_att() for _ in range(self.num_layers)])  # make_att_deprecated
        if self.is_encoder_decoder:
            self.enc_attention = nn.ModuleList([make_att() for _ in range(self.num_layers)])

        # Layer normalization for every residual layer
        self.ln = nn.ModuleList([nn.ModuleList([LayerNormalization(self.embed_dim)
                                                for _ in range(3 if self.is_encoder_decoder else 2)])
                                 for _ in range(self.num_layers)])
        self.ln_out = LayerNormalization(self.embed_dim)

        # Two-layer feed-forward with wide layer in the middle
        self.ff_1 = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.ff_dim),
                                                 nn.ReLU()) for _ in range(self.num_layers)])
        self.ff_2 = nn.ModuleList([nn.Linear(self.ff_dim, self.hidden_dim) for _ in range(self.num_layers)])

    def forward(self, states, masks, attention_bias):

        for ix in range(self.num_layers):
            new_states = self.ln[ix][0](states)
            new_states = self.attention[ix](states, states, masks, attention_bias)
            new_states = self.dropout(new_states)
            states = states + new_states

            new_states = self.ff_1[ix](self.ln[ix][1](states))
            new_states = self.dropout(new_states)
            new_states = self.ff_2[ix](new_states)
            new_states = self.dropout(new_states)
            states = states + new_states
        return self.ln_out(states)

    # Embed inputs. Note: applies scaling before positional encoding.
    def embed_inputs(self, inputs):
        # states = self.embed(inputs)
        states = inputs
        states = (states *
                  torch.tensor(states.shape[-1], dtype=torch.float32, device=next(self.parameters()).device).sqrt())
        states = states + self.pos_enc[:states.shape[1]].to(next(self.parameters()).device)[None, :, None]
        return states

    # Standard encoder-decoder attention, with dropout if training=True.
    # NOTE: tentatively does not support attention bias from the query domain to the key domain;
    # extending this should be straightforward.
    def enc_dec_attention(self, states, masks, key_states, key_masks, attention_bias):
        for ix in range(self.num_layers):
            new_states = self.ln[ix][0](states)
            new_states = self.attention[ix](new_states, new_states, masks, torch.zeros((0, 4), dtype=torch.int32))
            new_states = self.dropout(new_states)
            states = states + new_states

            new_states = self.ln[ix][2](states)
            new_states = self.enc_attention[ix](new_states, key_states, key_masks, attention_bias)
            new_states = self.dropout(new_states, )
            states = states + new_states

            new_states = self.ff_1[ix](self.ln[ix][1](states))
            new_states = self.dropout(new_states)
            new_states = self.ff_2[ix](new_states)
            new_states = self.dropout(new_states)
            states = states + new_states
        return self.ln_out(states)

    # Generates tokens from transformer states using the transposed embedding layer.
    def predict(self, states):
        return torch.matmul(states, self.embed.t())


class GreatEncoder(nn.Module):

    def __init__(self, config: GreatEncoderConfig, shared_embedding: Optional[nn.Embedding] = None):
        super(GreatEncoder, self).__init__()
        self.transformer = Transformer(model_config=config.transformer_config)
        self.vocab_size = config.vocab_size
        self.subtokens_per_token = config.subtokens_per_token

        if shared_embedding:
            self.token_embedding = shared_embedding
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.transformer.embed_dim)
        self.token_linear_down = nn.Linear(self.subtokens_per_token * self.transformer.embed_dim,
                                           self.transformer.embed_dim)

        if config.num_languages is not None:
            self.language_embedding = nn.Embedding(config.num_languages, self.transformer.embed_dim)
        else:
            self.language_embedding = None

        self._reset_parameters()

    def forward(self, input_tokens: torch.Tensor,
                edge_ixs,  # [edge_type, batch_index, source_index, target_index]
                attention_mask: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None,
                need_all_embeddings=False,
                languages: Optional[torch.Tensor] = None
                ):
        """
        :param input_tokens:
            the input token sequence
        :param edge_ixs:
        :param attention_mask: torch.Tensor [B, S, S]

        :param pad_mask: torch.Tensor [B, S]
            the pad mask where 1 indicates a PAD token
        :param need_all_embeddings:
        :return:
        """
        bsz, seq_len = input_tokens.shape[0], input_tokens.shape[1]

        token_embeddings = self.transformer.embed_inputs(self.token_embedding(input_tokens))
        token_embeddings = self.token_linear_down(token_embeddings.reshape(bsz, seq_len, -1))

        if self.language_embedding is not None:
            languages = languages.unsqueeze(-1)  # [B, 1]
            languages = self.language_embedding(languages)
            token_embeddings = token_embeddings + languages

        attention_mask = (attention_mask + pad_mask.unsqueeze(1) + pad_mask.unsqueeze(2)).clamp_max(1)

        transformer_output = self.transformer.forward(token_embeddings,
                                                      masks=attention_mask,
                                                      attention_bias=edge_ixs.t(),
                                                      )
        return transformer_output

    def forward_batch(self, batch: GreatBatch):
        return self.forward(input_tokens=batch.tokens,
                            edge_ixs=batch.edge_ixs,
                            attention_mask=batch.attention_mask,
                            pad_mask=batch.pad_mask,
                            languages=batch.languages)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
