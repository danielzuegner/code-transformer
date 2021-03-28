from collections import namedtuple
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_

from code_transformer.configuration.code_transformer import CodeTransformerCoreConfig
from code_transformer.modeling.code_transformer.distance_embeddings import TransformerPositionalEncoding

TransformerOutput = namedtuple("TransformerOutput",
                               ['out_emb', 'attentions', 'all_emb'])

TransformerLayerOutput = namedtuple("TransformerLayerOutput", ["content_stream_out", "query_stream_out",
                                                               "attentions"])
Attentions = namedtuple("Attentions", ["content_attention", "query_attention"])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError("unknown activation function")


class CodeTransformerLayer(TransformerEncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 num_relative_distances=1, use_token_distances=False, use_edge_embeddings=False,
                 use_content_content=True, use_content_pos=True,
                 use_pos_content=True, use_pos_pos=True, **kwargs):

        super(CodeTransformerLayer, self).__init__(d_model, nhead, dim_feedforward, dropout,
                                                   activation)
        self.d_model = d_model

        self.self_attn = RelativeMultiheadAttention(num_relative_distances, d_model, nhead,
                                                    use_token_distances=use_token_distances,
                                                    use_edge_embeddings=use_edge_embeddings,
                                                    use_content_content=use_content_content,
                                                    use_content_pos=use_content_pos,
                                                    use_pos_content=use_pos_content,
                                                    use_pos_pos=use_pos_pos,
                                                    dropout=dropout)

    def _reset_parameters(self):
        self.self_attn._reset_parameters()

    def post_attention(self, src: torch.Tensor, src2: torch.Tensor):
        """
        Feed-forward layer after the attention computations.

        Parameters
        ----------
        src: torch.Tensor, the input embeddings before the attention layer.
        src2: torch.Tensor, the output of the attention layer

        Returns
        -------
        torch.Tensor resulting from applying two feedforward layers and skip connection.

        """

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                query_stream: Optional[torch.Tensor] = None,
                attention_mask_query: Optional[torch.Tensor] = None,
                target_mapping: Optional[torch.Tensor] = None,
                relative_distances: Optional[Tuple[torch.Tensor]] = None,
                token_distances: Optional[torch.Tensor] = None,
                edge_indices: Optional[torch.Tensor] = None,
                edge_embeddings: Optional[torch.Tensor] = None,
                need_weights: bool = False, mems: Optional[torch.tensor] = None,
                asserts=True) -> TransformerLayerOutput:
        """
        Forward step of the code transformer layer.

        Parameters
        ----------
        src: torch.Tensor, dim [seq_len, batch_size, d_model]
            The content stream embeddings.
        src_mask: torch.Tensor, dim [seq_len, seq_len, batch_size, 1]
            Attention mask for the content stream.
        src_key_padding_mask: torch.Tensor (currently not used)
        query_stream: torch.tensor, dim [num_pedict, batch_size, d_model]
            The query stream embeddings.
        attention_mask_query: torch.Tensor, dim [seq_len, seq_len, batch_size, 1]
            Attention mask for the content stream (target nodes cannot see themselves).
        target_mapping: torch.Tensor, dim [num_predict, seq_len, batch_size]
            Mapping indicating which tokens are being predicted during pre-training.
            Entry [i,j,k] is 1 if token [j,k] is the i-th target.
        relative_distances: Tuple[torch.Tensor].
            Tuple containing the relative distances.
            The first element is the [num_distances, batch_size, seq_len, seq_len] dimensional index tensor, indexing
            the second element, which is the [num_distances, num_bins, batch_size, d_model] dimensional tensor
            containing the encoded distance bins.
            That is, the value at [d, b, i, j] of the index tensor is the index of the distance bin of the d-th distance
            of tokens i and j in batch b.
        token_distances: torch.Tensor, dim [num_token_distances, batch_size, d_model]
            The distances between tokens.
        edge_indices: torch.Tensor, dim [3, num_edges_in_batch]
        edge_embeddings: torch.Tensor, dim [num_edges_in_batch, d_model]
        need_weights: bool
            Whether to also return the attention probabilities.
        mems: torch.tensor (currently not used)
        asserts: bool
            Whether to verify dimensions via asserts.

        Returns
        -------
        outputs: tuple containing
        * the new content stream embeddings, dim [seq_len, batch_size, d_model]
        * the new query stream embeddings (or None if no query stream input is provided),
          dim [num_pedict, batch_size, d_model]
        * (Optional) if need_weights=True, tuple containing
            - content stream attention probabilities, dim [seq_len, seq_len, batch_size, num_head]
            - query stream attention probabilities, or None if no query stream input provided,
              dim [seq_len, seq_len, batch_size, num_head]
        """

        # two-stream attention with relative positional encoding
        if mems is not None:
            raise NotImplementedError("memory currently not implemented")
        content_stream_cat = src

        seq_len, bsz = src.shape[:2]
        if asserts:
            assert src.shape == (seq_len, bsz, self.d_model)
            if relative_distances is not None and len(relative_distances) > 0:
                assert relative_distances[0].shape == (self.self_attn.num_relative_distances
                                                       - int(self.self_attn.use_token_distances)
                                                       - int(self.self_attn.use_edge_embeddings), bsz, seq_len, seq_len)
                assert relative_distances[1].shape[0] == (self.self_attn.num_relative_distances
                                                          - int(self.self_attn.use_token_distances)
                                                          - int(self.self_attn.use_edge_embeddings))
                assert relative_distances[1].shape[-2:] == (bsz, self.d_model)
            if src_mask is not None:
                assert src_mask.shape == (seq_len, seq_len, bsz)
            if query_stream is not None:
                num_predict = query_stream.shape[0]
                assert query_stream.shape == (num_predict, bsz, self.d_model)
                if attention_mask_query is not None:
                    assert attention_mask_query.shape == (seq_len, seq_len, bsz)

        k_content_stream = F.linear(content_stream_cat, self.self_attn.get_k_proj_weight(),
                                    self.self_attn.get_k_proj_bias())
        v_content_stream = F.linear(content_stream_cat, self.self_attn.get_v_proj_weight(),
                                    self.self_attn.get_v_proj_bias())
        q_content_stream = F.linear(src, self.self_attn.get_q_proj_weight(), self.self_attn.get_q_proj_bias())

        k_position_encoding = None
        dist_ixs = None
        if relative_distances is not None:
            dist_ixs = relative_distances[0]
            encoded_distances = relative_distances[1]

            k_position_encoding = torch.einsum('rkbd,rhd->rkbh', encoded_distances, self.self_attn.get_r_proj_weight())
            k_position_encoding = k_position_encoding + self.self_attn.get_r_proj_bias()[:, None, None]

        k_token_pos_encoding = None
        if token_distances is not None:
            k_token_pos_encoding = torch.einsum('kbd,hd->kbh', token_distances,
                                                self.self_attn.get_r_token_proj_weight())
            k_token_pos_encoding = k_token_pos_encoding + self.self_attn.get_r_token_proj_bias()[None, None]

        k_edge_type_encoding = None
        if edge_embeddings is not None and self.self_attn.use_edge_embeddings:
            assert edge_indices is not None
            k_edge_type_encoding = torch.einsum('kd,hd->kh', edge_embeddings, self.self_attn.get_r_edge_proj_weight())
            k_edge_type_encoding = k_edge_type_encoding + self.self_attn.get_r_edge_proj_bias()[None]
        else:
            edge_indices = None
        # core attention ops for content stream
        att_out_content = self.self_attn.forward(
            q_content_stream, k_content_stream, v_content_stream, position_keys=k_position_encoding,
            token_pos_keys=k_token_pos_encoding,
            attn_mask=src_mask, distance_indices=dist_ixs, need_weights=need_weights,
            edge_embeddings=k_edge_type_encoding,
            edge_indices=edge_indices,
            key_padding_mask=src_key_padding_mask,
        )
        if need_weights:
            att_out_content, att_probs_content = att_out_content

        att_out_content = self.post_attention(src, att_out_content)

        if query_stream is not None:  # query stream attention (if query stream is provided)
            q_query_stream = F.linear(query_stream, self.self_attn.get_q_proj_weight(),
                                      self.self_attn.get_q_proj_bias())

            if target_mapping is not None:
                # initialize query stream by putting the targets' embeddings into dim [seq_len, bsz, d_model]
                # that is, only the embeddings of the target nodes are nonzero, all others are zero (along the
                # seq_len dimension).
                q_query_stream = torch.einsum('mbk,mlb->lbk', q_query_stream, target_mapping)

                att_out_query = self.self_attn.forward(
                    q_query_stream, k_content_stream, v_content_stream, position_keys=k_position_encoding,
                    token_pos_keys=k_token_pos_encoding,
                    attn_mask=attention_mask_query, distance_indices=dist_ixs, need_weights=need_weights,
                    edge_embeddings=k_edge_type_encoding,
                    edge_indices=edge_indices,
                    key_padding_mask=src_key_padding_mask,
                )

                if need_weights:
                    att_out_query, att_probs_query = att_out_query

                # filter the embeddings to only contain the targets' embeddings. All other rows in the first dimension
                # are discarded.
                att_out_query = torch.einsum('lbk,mlb->mbk', att_out_query, target_mapping)
            else:

                att_out_query = self.self_attn.forward(
                    q_query_stream, k_content_stream, v_content_stream, position_keys=k_position_encoding,
                    token_pos_keys=k_token_pos_encoding,
                    attn_mask=attention_mask_query, distance_indices=dist_ixs, need_weights=need_weights,
                    edge_embeddings=k_edge_type_encoding,
                    edge_indices=edge_indices,
                    key_padding_mask=src_key_padding_mask,
                )
                if need_weights:
                    att_out_query, att_probs_query = att_out_query
            att_out_query = self.post_attention(query_stream, att_out_query)
        else:
            att_out_query = None
            att_probs_query = None

        if not need_weights:
            att_probs_content = None
            att_probs_query = None

        outputs = TransformerLayerOutput(content_stream_out=att_out_content,
                                         query_stream_out=att_out_query,
                                         attentions=Attentions(content_attention=att_probs_content,
                                                               query_attention=att_probs_query)
                                         )
        return outputs


class RelativeMultiheadAttention(MultiheadAttention):

    def __init__(self, num_relative_distances: int,
                 embed_dim: int, num_heads: int,
                 use_token_distances: bool = False, use_edge_embeddings: bool = False,
                 use_content_content: bool = True, use_content_pos: bool = True, use_pos_content: bool = True,
                 use_pos_pos: bool = True, dropout: float = 0., bias: bool = True, add_bias_kv: bool = False,
                 add_zero_attn: bool = False, kdim: Optional[int] = None, vdim: Optional[int] = None):
        """

        Parameters
        ----------
        num_relative_distances: int
            The number of relative distances to consider.
        embed_dim: int
            The model embedding size.
        num_heads: int
            The number of attention heads.
        use_token_distances: bool, Default: False
            Whether to use the sequence token distances.
        use_edge_embeddings: bool, Default: False
            Whether to use edge embeddings.
        use_content_content: bool, Default: True
            Whether to use the content-content term in the computation of the attention scores.
        use_content_pos: bool: Default: True
            Whether to use the content-position term in the computation of the attention scores.
        use_pos_content: bool: Default: True
            Whether to use the position-content term in the computation of the attention scores.
        use_pos_pos: bool: Default: True
            Whether to use the position-position term in the computation of the attention scores.
        dropout: float: Default: 0.0
            The dropout rate.
        bias: bool
            Add bias as module parameter. Default: True. See also: torch.nn.MultiheadAttention
        add_bias_kv: bool
            add bias to the key and value sequences at dim=0. See also: torch.nn.MultiheadAttention
        add_zero_attn: bool
            add a new batch of zeros to the key and value sequences at dim=1. See also: torch.nn.MultiheadAttention
        kdim: int
            total number of features in key. Default: None. See also: torch.nn.MultiheadAttention
        vdim: int
            total number of features in key. Default: None. See also: torch.nn.MultiheadAttention
        """

        self.r_proj_weight = None
        self.r_proj_bias = None
        self.pos_pos_bias = None
        self.position_content_bias = None
        self.position_segment_bias = None
        self.use_token_distances = use_token_distances
        self.use_edge_embeddings = use_edge_embeddings
        super(RelativeMultiheadAttention, self).__init__(embed_dim, num_heads, dropout, bias, add_bias_kv,
                                                         add_zero_attn, kdim, vdim)
        if not self._qkv_same_embed_dim:
            raise ValueError("currently QKV need to have the same dimension")

        self.scale = 1 / (self.head_dim ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.num_relative_distances = num_relative_distances

        if not (self.num_relative_distances > 0 or self.use_token_distances or self.use_edge_embeddings):
            print("WARNING: num_relative_distances is 0 and no token distances or edge embeddings are used.")

        self.num_binned_distances = self.num_relative_distances
        self.num_relative_distances = (self.num_relative_distances + int(self.use_token_distances)
                                       + int(self.use_edge_embeddings))
        self.token_distances_ix = -(int(self.use_edge_embeddings) + 1) if self.use_token_distances else None
        self.edge_embeddings_ix = -1 if self.use_edge_embeddings else None

        self.use_content_content = torch.tensor(use_content_content, dtype=torch.float)
        self.use_content_pos = torch.tensor(use_content_pos, dtype=torch.float)
        self.use_pos_content = torch.tensor(use_pos_content, dtype=torch.float)
        self.use_pos_pos = torch.tensor(use_pos_pos, dtype=torch.float)

        self.r_proj_weight = nn.Parameter(torch.FloatTensor(self.num_relative_distances, self.embed_dim,
                                                            self.num_heads * self.head_dim))
        if bias:  # this is the bias of the projection, NOT the position-position bias (term (d) in the paper)
            self.r_proj_bias = nn.Parameter(torch.empty(self.num_relative_distances, self.embed_dim))
        else:
            self.r_proj_bias = torch.empty(self.num_relative_distances, self.embed_dim)

        self.register_parameter(f'pos_proj_weight', self.r_proj_weight)
        if bias:
            self.register_parameter(f'pos_proj_bias', self.r_proj_bias)

        self.pos_pos_bias = nn.Parameter(torch.FloatTensor(self.num_relative_distances, self.num_heads * self.head_dim))
        self.register_parameter(f'pos_pos_bias', self.pos_pos_bias)

        self.position_segment_bias = nn.Parameter(torch.FloatTensor(self.num_heads, self.head_dim))
        self.position_content_bias = nn.Parameter(torch.FloatTensor(self.num_heads, self.head_dim))

        self._reset_parameters()

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))

        # x = x[:, :, :, :klen]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: Optional[bool] = True,
                attn_mask: Optional[torch.Tensor] = None, position_keys: Optional[torch.Tensor] = None,
                token_pos_keys: Optional[torch.Tensor] = None,
                distance_indices: Optional[torch.Tensor] = None,
                edge_indices: Optional[torch.Tensor] = None,
                edge_embeddings: Optional[torch.Tensor] = None,
                ):
        """
        Core operations of the relative attention module.

        Parameters
        ----------
        query: torch.Tensor, dim [tgt, batch, n_head * dim_head]
        key: torch.Tensor, dim [seq, batch, n_head * dim_head]
        value: torch.Tensor, dim [seq, batch, n_head * dim_head]
        key_padding_mask: torch.Tensor, shape [seq, batch]
            Binary padding mask. 1 indicates that an item is padded.
        position_keys: torch.Tensor, dim [num_relative_distances, num_distance_bins, batch, n_head * dim_head]
            The key representations of the unique relative distance values.
        token_pos_keys: torch.Tensor, dim [num_token_distances, batch, nhead * dim_head]
            The key representations of the sequence token distances.
        attn_mask: torch.Tensor, dim [tgt, seq, batch]:
            Binary attention mask. 1 indicates that a token cannot attend the other one.
        distance_indices: torch.Tensor, dim [num_relative_distances, batch, tgt, seq]
        need_weights: bool (whether to return the attention probabilities)

        edge_indices: torch.Tensor, dim [3, num_edges_in_batch]
        edge_embeddings: torch.Tensor, dim [num_edges_in_batch, n_head*dim_head]

        Returns
        -------
        attention_output: torch.Tensor [tgt, batch, num_head * head_dim]
        optional: attention_prob, torch.Tensor [batch, tgt, seq, num_head]

        """
        if token_pos_keys is None and self.use_token_distances:
            raise ValueError("use_token_distances was True but no token pos keys provided.")
        if edge_embeddings is None and self.use_edge_embeddings:
            raise ValueError("use_token_distances was True but no edge embeddings provided.")

        seq_len = key.shape[0]
        tgt_len, bsz = query.shape[:2]
        query_cp_bias = self.use_content_content * query + self.use_pos_content * self.position_content_bias.reshape(-1)
        query_cp_bias = query_cp_bias.contiguous().view(tgt_len, bsz * self.num_heads,
                                                        self.head_dim).transpose(0, 1)

        key = key.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        ac = torch.bmm(query_cp_bias, key.transpose(1, 2))
        assert list(ac.size()) == [bsz * self.num_heads, tgt_len, seq_len]

        query_pp_bias = self.use_content_pos * query[None] + self.use_pos_pos * self.pos_pos_bias[:, None, None]
        query_pp_bias = query_pp_bias.reshape([self.num_relative_distances, tgt_len, bsz,
                                               self.num_heads, self.head_dim])

        bd = torch.zeros((bsz, self.num_heads, tgt_len, seq_len)).to(ac.device)
        if position_keys is not None:
            q_pp_bias = query_pp_bias[:self.num_binned_distances]

            position_keys = position_keys.reshape([self.num_binned_distances, -1, bsz, self.num_heads, self.head_dim])
            dist_bd = torch.einsum('ribnd,rjbnd->rbnij', q_pp_bias, position_keys)

            sp_expand = distance_indices.long().unsqueeze(2).expand([-1, -1, dist_bd.shape[2], -1, -1])
            dist_bd = dist_bd.gather(-1, sp_expand)
            bd += dist_bd.sum(0)  # sum over num_relative_distances

        if token_pos_keys is not None:
            token_pos_keys = token_pos_keys.reshape([-1, 1, self.num_heads, self.head_dim])
            token_pos_keys = token_pos_keys.expand(-1, bsz, -1, -1)
            dist_bd = torch.einsum('ibnd,jbnd->bnij', query_pp_bias[self.token_distances_ix], token_pos_keys)
            dist_bd = self.rel_shift_bnij(dist_bd, klen=seq_len)
            bd += dist_bd

        if edge_indices is not None:
            assert edge_embeddings is not None
            edge_emb_keys = edge_embeddings.reshape([1, -1, self.num_heads, self.head_dim])
            qpp = query_pp_bias[self.edge_embeddings_ix]
            qpp = qpp.transpose(0, 1)
            qpp_sel = qpp[tuple(edge_indices[:2])]
            qpp_sel = qpp_sel.unsqueeze(0)
            edge_att = torch.einsum('rend,rend->en', qpp_sel, edge_emb_keys)

            bd = bd.permute((0, 2, 3, 1))  # b, tgt, seq, num_head
            bd[tuple(edge_indices)] += edge_att
            bd = bd.permute((0, 3, 1, 2))  # b, num_head, tgt, seq

        attention_raw = (ac.view_as(bd) + bd) * self.scale

        if key_padding_mask is not None:
            key_padding = True
            key_padding_mask = key_padding_mask.unsqueeze(0)
            assert key_padding_mask.shape == (1, seq_len, bsz)
        else:
            key_padding = False
            key_padding_mask = torch.zeros((1, seq_len, bsz)).to(next(self.parameters()).device)

        if attn_mask is not None:
            attn = True
            assert attn_mask.shape == (tgt_len, seq_len, bsz)
            attn_mask = key_padding_mask + attn_mask
        else:
            attn = False
            attn_mask = key_padding_mask.expand(tgt_len, -1, -1)

        if key_padding or attn:
            attn_mask = (attn_mask > 0).to(attn_mask.dtype)
            if edge_indices is not None:
                attn_mask = attn_mask.permute((2, 0, 1))
                attn_mask[tuple(edge_indices)] = 0
                attn_mask = attn_mask.permute((1, 2, 0))

            if attn_mask.dtype == torch.float16:
                attention_raw = attention_raw - 65500 * torch.einsum('ijbn->bnij', attn_mask.unsqueeze(-1))
            else:
                attention_raw = attention_raw - 1e30 * torch.einsum('ijbn->bnij', attn_mask.unsqueeze(-1))

        attention_prob = F.softmax(attention_raw, dim=3)
        # attention_prob = self.dropout(attention_prob) # dropout here can lead to attention scores > 1!

        attention_output = torch.einsum('bnij,jbnd->ibnd', attention_prob, value.view(seq_len, bsz, self.num_heads,
                                                                                      self.head_dim))

        attention_output = attention_output.reshape([tgt_len, bsz, self.num_heads * self.head_dim])

        attention_output = F.linear(attention_output, self.out_proj.weight,
                                    self.out_proj.bias)

        if need_weights:
            return attention_output, attention_prob.permute((0, 2, 3, 1))

        return attention_output

    def proj_bias(self, which="q"):
        if which == "q":
            _start = 0
            _end = self.embed_dim
        elif which == "k":
            _start = self.embed_dim
            _end = self.embed_dim * 2
        elif which == "v":
            _start = self.embed_dim * 2
            _end = None
        else:
            raise ValueError("unknown projection weight")

        if self.in_proj_bias is not None:
            proj_bias = self.in_proj_bias[_start:_end]
        else:
            proj_bias = torch.tensor(0.0, dtype=torch.float).to(self.in_proj_weight)
        return proj_bias

    def proj_weight(self, which="q"):
        if which == "q":
            _start = 0
            _end = self.embed_dim
        elif which == "k":
            _start = self.embed_dim
            _end = self.embed_dim * 2
        elif which == "v":
            _start = self.embed_dim * 2
            _end = None
        else:
            raise ValueError("unknown projection weight")

        proj_weight = self.in_proj_weight[_start:_end, :]

        return proj_weight

    def get_q_proj_weight(self):
        return self.proj_weight(which="q")

    def get_k_proj_weight(self):
        return self.proj_weight(which="k")

    def get_v_proj_weight(self):
        return self.proj_weight(which="v")

    def get_r_proj_weight(self):
        return self.r_proj_weight[:self.num_binned_distances]

    def get_r_token_proj_weight(self):
        assert self.use_token_distances, "model does not use token distances."
        return self.r_proj_weight[self.token_distances_ix]

    def get_r_edge_proj_weight(self):
        assert self.use_edge_embeddings, "model does not use edge embeddings."
        return self.r_proj_weight[self.edge_embeddings_ix]

    def get_q_proj_bias(self):
        return self.proj_bias(which="q")

    def get_k_proj_bias(self):
        return self.proj_bias(which="k")

    def get_v_proj_bias(self):
        return self.proj_bias(which="v")

    def get_r_proj_bias(self):
        return self.r_proj_bias[:self.num_binned_distances]

    def get_r_token_proj_bias(self):
        assert self.use_token_distances, "model does not use token distances."
        return self.r_proj_bias[self.token_distances_ix]

    def get_r_edge_proj_bias(self):
        assert self.use_edge_embeddings, "model does not use edge embeddings."
        return self.r_proj_bias[self.edge_embeddings_ix]

    def _reset_parameters(self):
        super(RelativeMultiheadAttention, self)._reset_parameters()
        if self.r_proj_weight is not None:
            xavier_uniform_(self.r_proj_weight)

        if self.r_proj_bias is not None:
            constant_(self.r_proj_bias, 0.)

        xavier_normal_(self.position_segment_bias) if self.position_segment_bias is not None else None
        xavier_normal_(self.position_content_bias) if self.position_content_bias is not None else None
        if self.pos_pos_bias is not None:
            xavier_normal_(self.pos_pos_bias)


class CodeTransformer(TransformerEncoder):
    def __init__(self, config: CodeTransformerCoreConfig):
        if not isinstance(config.encoder_layer, CodeTransformerLayer):
            encoder_layer = CodeTransformerLayer(**config.encoder_layer)
        else:
            encoder_layer = config.encoder_layer
        super(CodeTransformer, self).__init__(encoder_layer, config.num_layers, config.norm)

        self.use_token_distances = encoder_layer.self_attn.use_token_distances
        self.d_model = encoder_layer.d_model
        self.query_stream_emb = nn.Parameter(torch.FloatTensor(1, 1, self.d_model))
        self.dropout = nn.Dropout(p=encoder_layer.dropout.p)

        if 'positional_encoding' not in config or config.positional_encoding is None:
            self.positional_embedding = TransformerPositionalEncoding(d_model=self.d_model)
        else:
            encoding_type = None
            if "positional_encoding_type" in config.positional_encoding:
                encoding_type = config.positional_encoding['positional_encoding_type']

            if encoding_type is None or encoding_type == "Transformer":
                self.positional_embedding = TransformerPositionalEncoding(**config.positional_encoding,
                                                                          d_model=self.d_model)
            else:
                raise NotImplementedError(f"Unknown positional encoding: {self.positional_encoding_type}")

        self._reset_parameters()

    @staticmethod
    def from_old(encoder_layer: CodeTransformerLayer, num_layers, norm=None, pos_emb_base_pow=10000):
        return CodeTransformer(CodeTransformerCoreConfig(encoder_layer, num_layers, norm=norm,
                                                         positional_encoding=dict(pos_emb_base_pow=pos_emb_base_pow)))

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for l in self.layers:
            l._reset_parameters()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                relative_distances: Optional[List[Tuple[torch.Tensor]]] = None,
                edge_embeddings: Optional[Tuple[torch.Tensor]] = None,
                target_mapping: Optional[torch.Tensor] = None,
                need_weights: Optional[bool] = False,
                need_all_embeddings: Optional[bool] = False) -> TransformerOutput:
        """
        Forward pass of the Code Transformer.

        Parameters
        ----------
        src: torch.Tensor, dim [batch_size, seq_len, d_model]
            The input embeddings.
        src_mask: torch.Tensor, dim [batch_size, seq_len, seq_len]
            Attention mask where 1 indicates that a token cannot attend the other token.
        src_key_padding_mask: torch.Tensor, dim [batch_size, seq_len]
            Padding mask where 1 indicates that a token is padded. Padded tokens cannot be attended by any token.
        relative_distances: list of tuples of torch.Tensor
            The relative distances between the tokens. Each tuple contains two tensors:
                * The [batch_size, seq_len, seq_len] dimensional tensor indexing the second tensor, which is
                * The [num_distance_bins, batch_size] dimensional tensor containing the values of the distance bins.
        edge_embeddings: tuple of torch.Tensor
            * edge_embeddings, dtype float [num_edges_in_batch, d_model]
            * edge_indices, dtype int [3, num_edges_in_batch]
        target_mapping: torch.Tensor, dim [batch_size, num_predict, seq_len]
            Mapping indicating which tokens are being predicted during pre-training.
            Entry [i,j,k] is 1 if token k is the j-th target in batch i.
        need_weights: bool (whether to return the attention probabilities)
        need_all_embeddings: bool (whether to return the embeddings at every layer)

        Returns
        -------
        outputs: tuple containing
            * output embeddings: torch.Tensor dim [batch_size, num_predict, d_model] if target_mapping is provided,
              else [batch_size, seq_len, d_model]
            * Optional: if need_weights is True: attentions, list with length n_layers. Each entry is a tuple containing
                * Content stream embeddings: torch.Tensor dim [batch_size, seq_len, seq_len, num_heads]
                * Query stream embeddings (if target_mapping is provided, else None),
                  dim [batch_size, seq_len, seq_len, num_heads]
        """
        if src_key_padding_mask is not None:
            assert (src_key_padding_mask[:, 0] == 0).all(), "First token padded; check padding mask!"
        content_stream_out = self.dropout(src.transpose(0, 1).contiguous())
        src_mask = src_mask.permute(1, 2, 0).contiguous() if src_mask is not None else None

        src_key_padding_mask = (src_key_padding_mask.transpose(0, 1).contiguous()
                                if src_key_padding_mask is not None else None)

        target_mapping = target_mapping.permute(1, 2, 0).contiguous() if target_mapping is not None else None

        seq_len, bsz = content_stream_out.shape[0], content_stream_out.shape[1]

        if edge_embeddings is not None:
            edge_indices, edge_embeddings = edge_embeddings
        else:
            edge_embeddings = None
            edge_indices = None

        # Query stream attention mask: tokens cannot see themselves (as reflected by the permutation mask)
        if src_mask is not None:
            attn_mask_q = src_mask
            attn_mask_q = (attn_mask_q > 0).to(src)
            # Content stream attention mask: tokens can see themselves, so we set the diagonal to zero.
            attn_mask_h = -torch.eye(seq_len).to(attn_mask_q)
            attn_mask_h = ((attn_mask_q + attn_mask_h[:, :, None]) > 0).to(attn_mask_q)
        else:
            attn_mask_q = None
            attn_mask_h = None

        encoded_token_distances = None
        if self.use_token_distances:
            possible_distances = torch.arange(seq_len, -seq_len, -1.0).unsqueeze(1).to(next(self.parameters()).device)
            encoded_token_distances = self.positional_embedding(possible_distances)

        if relative_distances is not None and len(relative_distances) > 0:
            encoded_distances = []
            for distance_indices, possible_distances in relative_distances:
                pos_emb = self.positional_embedding(possible_distances)
                pos_emb = self.dropout(pos_emb)
                encoded_distances.append((distance_indices, pos_emb))
            distance_indices = torch.stack([x[0] for x in encoded_distances])
            distance_encodings = torch.stack([x[1] for x in encoded_distances])
            relative_distances = (distance_indices, distance_encodings)
        else:
            relative_distances = None

        if target_mapping is not None:
            query_stream_out = self.query_stream_emb.expand(target_mapping.shape[0], bsz, -1)
            query_stream_out = self.dropout(query_stream_out)
        else:
            query_stream_out = None

        attentions = []
        embeddings = []
        if need_all_embeddings:
            embeddings.append((content_stream_out, query_stream_out))
        for mod in self.layers:
            outputs = mod(content_stream_out, src_mask=attn_mask_h, attention_mask_query=attn_mask_q,
                          src_key_padding_mask=src_key_padding_mask,
                          relative_distances=relative_distances, token_distances=encoded_token_distances,
                          target_mapping=target_mapping,
                          query_stream=query_stream_out, need_weights=need_weights,
                          edge_embeddings=edge_embeddings, edge_indices=edge_indices
                          )
            content_stream_out = outputs.content_stream_out
            query_stream_out = outputs.query_stream_out
            # content_stream_out, query_stream_out = outputs[:2]
            if need_weights:
                attentions.append(outputs[2])
            if need_all_embeddings:
                embeddings.append((content_stream_out, query_stream_out))

        output = self.dropout(query_stream_out if query_stream_out is not None else content_stream_out)

        if not need_weights:
            attentions = None
        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        outputs = TransformerOutput(out_emb=output.permute(1, 0, 2).contiguous(),
                                    attentions=attentions, all_emb=embeddings)

        return outputs
