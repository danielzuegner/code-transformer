from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from transformers import XLNetModel, XLNetConfig

from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.code_transformer.code_transformer import TransformerOutput
from code_transformer.modeling.code_transformer.lm import CodeTransformerOutput
from code_transformer.preprocessing.datamanager.base import CTBatch


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


class XLNetLMEncoder(XLNetModel):

    def __init__(self, config: TransformerLMEncoderConfig):

        if isinstance(config.transformer, XLNetConfig):
            super(XLNetLMEncoder, self).__init__(config.transformer)
        else:
            super(XLNetLMEncoder, self).__init__(XLNetConfig(**config.transformer))

        self.vocab_size = config.vocab_size
        self.subtokens_per_token = config.subtokens_per_token
        self.num_token_types = config.num_token_types

        downproj_add_dimension = 0
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        if self.num_token_types is not None:
            self.token_type_embedding = nn.Embedding(self.num_token_types, self.d_model)
            downproj_add_dimension += 1

        self.token_linear = nn.Linear((self.subtokens_per_token + downproj_add_dimension) * self.d_model, self.d_model)
        self.token_linear_up = nn.Linear(self.d_model, self.subtokens_per_token * self.d_model)

        self.input_nonlinearity = None
        self.output_nonlinearity = None

        if config.input_nonlinearity is not None:
            self.input_nonlinearity = _get_activation_fn(config.input_nonlinearity)

        if config.num_languages is not None:
            self.language_embedding = nn.Embedding(config.num_languages, self.d_model)
        else:
            self.language_embedding = None

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self,
            input_ids=None,
            pad_mask=None,
            mems=None,
            attention_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            languages=None,
            need_all_embeddings=False
    ):
        """
        :param input_ids:
            the input token sequence
        :param pad_mask:
            the pad mask where 1 indicates that a position is a PAD token
        :param mems:
        :param attention_mask:
        :param target_mapping:
        :param token_type_ids:
        :param input_mask:
        :param head_mask:
        :param inputs_embeds:
        :param languages:
        :param need_all_embeddings:
        :return:
        """
        # attention_mask: 1 indicates tokens that are NOT MASKED, 0 indicates tokens that are padded
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            emb_cat = torch.cat((token_embeddings, token_type_embeddings.unsqueeze(2)), dim=-2).reshape(
                [bsz, seq_len, -1])
        else:
            emb_cat = torch.cat((token_embeddings,), dim=-2).reshape([bsz, seq_len, -1])

        token_embeddings = self.token_linear(emb_cat)

        if self.language_embedding is not None:
            languages = languages.unsqueeze(-1)  # [B, 1]
            languages = self.language_embedding(languages)
            token_embeddings = token_embeddings + languages

        if self.input_nonlinearity is not None:
            token_embeddings = self.input_nonlinearity(token_embeddings)

        transformer_output = super(XLNetLMEncoder, self).forward(input_ids=None, inputs_embeds=token_embeddings,
                                                                 target_mapping=target_mapping,
                                                                 perm_mask=attention_mask,
                                                                 attention_mask=1 - pad_mask, mems=mems,
                                                                 head_mask=head_mask,
                                                                 output_hidden_states=need_all_embeddings,
                                                                 return_dict=True)

        output = transformer_output.last_hidden_state
        attentions = transformer_output.attentions

        all_emb = None
        if need_all_embeddings:
            all_emb = transformer_output.hidden_states
            all_emb = list(zip([content_stream.transpose(0, 1) for content_stream in all_emb[0::2]],
                               [query_stream.transpose(0, 1) for query_stream in all_emb[1::2]]))

        outputs = TransformerOutput(out_emb=output, attentions=attentions, all_emb=all_emb)
        return outputs


class XLNetLMEncoderLSTMAdapter(nn.Module):
    def __init__(self, xl_net_lm_encoder: XLNetLMEncoder):
        super(XLNetLMEncoderLSTMAdapter, self).__init__()
        self.xl_net_lm_encoder = xl_net_lm_encoder
        self.d_model = xl_net_lm_encoder.d_model
        self.vocab_size = xl_net_lm_encoder.vocab_size
        self.token_embedding = xl_net_lm_encoder.token_embedding

    def forward(self, input_tokens: torch.Tensor, input_node_types: torch.Tensor,
                relative_distances: List[Tuple[torch.Tensor]],
                input_token_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None,
                target_mapping: Optional[torch.Tensor] = None,
                need_weights: Optional[bool] = False) -> TransformerOutput:
        return self.xl_net_lm_encoder.forward(input_ids=input_tokens,
                                              token_type_ids=input_token_types, attention_mask=pad_mask,
                                              perm_mask=attention_mask, target_mapping=target_mapping)


class XLNetLanguageModel(nn.Module):

    def __init__(self, lm_encoder: XLNetLMEncoder,
                 output_nonlinearity=None, loss_fct=nn.CrossEntropyLoss(ignore_index=-1),
                 output_sub_tokens_per_token=5):
        super(XLNetLanguageModel, self).__init__()

        self.lm_encoder = lm_encoder
        self.output_sub_tokens_per_token = output_sub_tokens_per_token
        self.token_linear_up = nn.Linear(lm_encoder.d_model, output_sub_tokens_per_token * lm_encoder.d_model)
        self.token_embedding = lm_encoder.token_embedding
        self.loss_fct = loss_fct

        if output_nonlinearity is not None:
            self.output_nonlinearity = _get_activation_fn(output_nonlinearity)
        else:
            self.output_nonlinearity = None

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            mems=None,
            perm_mask=None,
            target_mapping=None,
            token_type_ids=None,
            input_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None
    ):

        transformer_output = self.lm_encoder.forward(input_ids=input_ids, attention_mask=attention_mask, mems=mems,
                                                     perm_mask=perm_mask, target_mapping=target_mapping,
                                                     token_type_ids=token_type_ids, input_mask=input_mask,
                                                     head_mask=head_mask, inputs_embeds=inputs_embeds)

        output_emb = transformer_output[0]
        output_up = self.token_linear_up(output_emb).reshape([output_emb.shape[0], output_emb.shape[1],
                                                              output_emb.shape[2], -1])
        if self.output_nonlinearity is not None:
            output_up = self.output_nonlinearity(output_up)

        output_up = output_up.transpose(2, 3)
        logits = output_up @ self.token_embedding.weight.T

        if labels is not None:
            # Flatten the tokens
            loss = self.loss_fct(logits.view(-1, logits.size(-1)),
                                 labels.view(-1))
        else:
            loss = None

        attentions = None
        if self.lm_encoder.config.output_attentions:
            attentions = transformer_output[-1]

        outputs = CodeTransformerOutput(loss=loss, logits=logits, attentions=attentions)
        return outputs

    def forward_batch(self, batch: CTBatch) -> CodeTransformerOutput:
        output = self.forward(input_ids=batch.tokens, token_type_ids=batch.token_types,
                              attention_mask=batch.pad_mask,
                              perm_mask=batch.perm_mask, target_mapping=batch.target_mapping, labels=batch.labels, )
        return output
