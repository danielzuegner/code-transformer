from typing import Union, List, Tuple, Optional, NamedTuple

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from code_transformer.configuration.code_transformer import CodeTransformerCoreConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.code_transformer.code_transformer import _get_activation_fn, \
    CodeTransformer, TransformerOutput
from code_transformer.preprocessing.datamanager.base import CTBatch


class CodeTransformerOutput(NamedTuple):
    loss: torch.Tensor
    logits: torch.Tensor
    attentions: Optional[torch.Tensor]

    # Pointer mechanism
    pointer_gates: Optional[torch.Tensor] = None
    pointer_attentions: Optional[torch.Tensor] = None
    pointer_attention_distributions: Optional[torch.Tensor] = None

    def cpu(self):
        cpu_loss = self.loss.cpu()
        cpu_logits = self.logits.detach().cpu()
        cpu_attentions = None if self.attentions is None else self.attentions.detach().cpu()
        pointer_gates = None if self.pointer_gates is None else self.pointer_gates.detach().cpu()
        pointer_attentions = None if self.pointer_attentions is None else self.pointer_attentions.detach().cpu()

        return CodeTransformerOutput(cpu_loss, cpu_logits, cpu_attentions,
                                     pointer_gates=pointer_gates, pointer_attentions=pointer_attentions)


class TransformerLMEncoder(nn.Module):

    def __init__(self, config: TransformerLMEncoderConfig):
        super(TransformerLMEncoder, self).__init__()

        if not isinstance(config.transformer, CodeTransformer):
            self.transformer = CodeTransformer(CodeTransformerCoreConfig(**config.transformer))
        else:
            self.transformer = config.transformer
        self.vocab_size = config.vocab_size
        self.subtokens_per_token = config.subtokens_per_token
        self.num_node_types = config.num_node_types
        self.num_token_types = config.num_token_types
        self.d_model = self.transformer.d_model

        downproj_add_dimension = 1
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.node_type_embedding = nn.Embedding(self.num_node_types, self.d_model)
        if self.num_token_types is not None:
            self.token_type_embedding = nn.Embedding(self.num_token_types, self.d_model)
            downproj_add_dimension += 1

        self.token_linear = nn.Linear((self.subtokens_per_token + downproj_add_dimension) * self.d_model, self.d_model)

        self.input_nonlinearity = None

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

    def forward(self, input_tokens: torch.Tensor, input_node_types: torch.Tensor,
                relative_distances: List[Tuple[torch.Tensor]],
                input_token_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None,
                target_mapping: Optional[torch.Tensor] = None,
                languages: Optional[torch.Tensor] = None,
                need_weights: Optional[bool] = False,
                need_all_embeddings: Optional[bool] = False) -> TransformerOutput:
        """
        Forward step of the Code Transformer Language Model.

        Parameters
        ----------
        input_tokens: torch.LongTensor, dim [batch_size, seq_len, num_subtokens]
            Tensor containing the vocabulary IDs of the subtokens in the sequence.
        input_node_types: torch.LongTensor, dim [batch_size, seq_len]
            Tensor containing the AST node types in the sequence.
        input_token_types: Optional[torch.Tensor]. Dim [batch_size, seq_len]
            Tensor containing the parsed token types.
        relative_distances: list of tuples of torch.Tensor
            The relative distances between the tokens. Each tuple contains two tensors:
                * The [batch_size, seq_len, seq_len] dimensional tensor indexing the second tensor, which is
                * The [num_distance_bins, batch_size] dimensional tensor containing the values of the distance bins.
        attention_mask: (Optional) torch.Tensor, dim [batch_size, seq_len, seq_len]
            Attention mask where 1 indicates that a token cannot attend the other token.
        pad_mask: (Optional) torch.Tensor, dim [batch_size, seq_len]
            Padding mask where 1 indicates that a token is padded. Padded tokens cannot be attended by any token.
        target_mapping: (Optional) torch.Tensor, dim [batch_size, num_predict, seq_len]
            Mapping indicating which tokens are being predicted during pre-training.
            Entry [i,j,k] is 1 if token k is the j-th target in batch i.
        need_weights: (Optional) bool (whether to return the attention probabilities)

        Returns
        -------
        output: tuple containing
            * (Optional) classification loss, torch.Tensor (scalar)
            * logits, torch.Tensor dim [batch_size, num_predict, n_vocabulary] if target_mapping is provided,
              else [batch_size, seq_len, n_vocabulary]
            * Optional: if need_weights is True: attentions, list with length n_layers. Each entry is a tuple containing
                * Content stream embeddings: torch.Tensor dim [batch_size, seq_len, seq_len, num_heads]
                * Query stream embeddings (if target_mapping is provided, else None),
                  dim [batch_size, seq_len, seq_len, num_heads]
        """

        bsz, seq_len = input_tokens.shape[0], input_tokens.shape[1]
        token_embeddings = self.token_embedding(input_tokens)
        node_types_mapped = input_node_types
        node_embeddings = self.node_type_embedding(node_types_mapped)

        if input_token_types is not None:
            token_type_embeddings = self.token_type_embedding(input_token_types)
            emb_cat = torch.cat((token_embeddings, node_embeddings.unsqueeze(2),
                                 token_type_embeddings.unsqueeze(2)), dim=-2).reshape([bsz, seq_len, -1])
        else:
            emb_cat = torch.cat((token_embeddings, node_embeddings.unsqueeze(2)), dim=-2).reshape([bsz, seq_len, -1])
        token_embeddings = self.token_linear(emb_cat)

        if self.language_embedding is not None:
            languages = languages.unsqueeze(-1)  # [B, 1]
            languages = self.language_embedding(languages)
            token_embeddings = token_embeddings + languages

        if self.input_nonlinearity is not None:
            token_embeddings = self.input_nonlinearity(token_embeddings)
        # token_embeddings = token_embeddings.transpose(0, 1)  # [seq, batch, dim] => [batch, seq, dim]

        transformer_output = self.transformer.forward(token_embeddings,
                                                      target_mapping=target_mapping,
                                                      need_weights=need_weights,
                                                      src_mask=attention_mask,
                                                      src_key_padding_mask=pad_mask,
                                                      relative_distances=relative_distances,
                                                      need_all_embeddings=need_all_embeddings)
        return transformer_output

    def forward_batch(self, batch: CTBatch, need_weights=False, need_all_embeddings=False) -> \
            TransformerOutput:
        output = self.forward(batch.tokens, input_node_types=batch.node_types, input_token_types=batch.token_types,
                              relative_distances=batch.relative_distances, attention_mask=batch.perm_mask,
                              pad_mask=1 - batch.pad_mask, target_mapping=batch.target_mapping,
                              need_weights=need_weights, languages=batch.languages,
                              need_all_embeddings=need_all_embeddings)
        return output


class TransformerLanguageModel(nn.Module):

    def __init__(self, transformer_lm_encoder: Union[TransformerLMEncoder, TransformerLMEncoderConfig],
                 output_nonlinearity=None, loss_fct=nn.CrossEntropyLoss(ignore_index=-1), **kwargs):

        super(TransformerLanguageModel, self).__init__()
        if not isinstance(transformer_lm_encoder, TransformerLMEncoder):
            self.transformer_lm_encoder = TransformerLMEncoder(TransformerLMEncoderConfig(**transformer_lm_encoder))
        else:
            self.transformer_lm_encoder = transformer_lm_encoder

        self.d_model = self.transformer_lm_encoder.d_model
        self.token_linear_up = nn.Linear(self.d_model,
                                         self.transformer_lm_encoder.subtokens_per_token * self.d_model)

        self.output_nonlinearity = None

        if output_nonlinearity is not None:
            self.output_nonlinearity = _get_activation_fn(output_nonlinearity)
        self.loss_fct = loss_fct

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, input_tokens: torch.Tensor, input_node_types: torch.Tensor,
                relative_distances: List[Tuple[torch.Tensor]],
                input_token_types: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None,
                target_mapping: Optional[torch.Tensor] = None,
                need_weights: Optional[bool] = False, labels: Optional[torch.Tensor] = None) -> CodeTransformerOutput:
        """
        Forward step of the Code Transformer Language Model.

        Parameters
        ----------
        input_tokens: torch.LongTensor, dim [batch_size, seq_len, num_subtokens]
            Tensor containing the vocabulary IDs of the subtokens in the sequence.
        input_node_types: torch.LongTensor, dim [batch_size, seq_len]
            Tensor containing the AST node types in the sequence.
        input_token_types: Optional[torch.Tensor]. Dim [batch_size, seq_len]
            Tensor containing the parsed token types.
        relative_distances: list of tuples of torch.Tensor
            The relative distances between the tokens. Each tuple contains two tensors:
                * The [batch_size, seq_len, seq_len] dimensional tensor indexing the second tensor, which is
                * The [num_distance_bins, batch_size] dimensional tensor containing the values of the distance bins.
        attention_mask: (Optional) torch.Tensor, dim [batch_size, seq_len, seq_len]
            Attention mask where 1 indicates that a token cannot attend the other token.
        pad_mask: (Optional) torch.Tensor, dim [batch_size, seq_len]
            Padding mask where 1 indicates that a token is padded. Padded tokens cannot be attended by any token.
        target_mapping: (Optional) torch.Tensor, dim [batch_size, num_predict, seq_len]
            Mapping indicating which tokens are being predicted during pre-training.
            Entry [i,j,k] is 1 if token k is the j-th target in batch i.
        need_weights: (Optional) bool (whether to return the attention probabilities)
        labels: (Optional) torch.Tensor dim [batch_size, num_predict, num_subtokens]
            The IDs of the target subtokens.

        Returns
        -------
        output: tuple containing
            * (Optional) classification loss, torch.Tensor (scalar)
            * logits, torch.Tensor dim [batch_size, num_predict, n_vocabulary] if target_mapping is provided,
              else [batch_size, seq_len, n_vocabulary]
            * Optional: if need_weights is True: attentions, list with length n_layers. Each entry is a tuple containing
                * Content stream embeddings: torch.Tensor dim [batch_size, seq_len, seq_len, num_heads]
                * Query stream embeddings (if target_mapping is provided, else None),
                  dim [batch_size, seq_len, seq_len, num_heads]
        """

        transformer_output = self.transformer_lm_encoder.forward(input_tokens=input_tokens,
                                                                 input_node_types=input_node_types,
                                                                 relative_distances=relative_distances,
                                                                 input_token_types=input_token_types,
                                                                 attention_mask=attention_mask,
                                                                 pad_mask=pad_mask, target_mapping=target_mapping,
                                                                 need_weights=need_weights)
        output_emb = transformer_output[0]
        output_up = self.token_linear_up(output_emb).reshape([output_emb.shape[0], output_emb.shape[1],
                                                              output_emb.shape[2], -1])
        if self.output_nonlinearity is not None:
            output_up = self.output_nonlinearity(output_up)

        output_up = output_up.transpose(2, 3)
        logits = output_up @ self.transformer_lm_encoder.token_embedding.weight.T

        if labels is not None:
            # Flatten the tokens
            loss = self.loss_fct(logits.view(-1, logits.size(-1)),
                                 labels.view(-1))
        else:
            loss = None

        outputs = CodeTransformerOutput(loss=loss,
                                        logits=logits,
                                        attentions=transformer_output.attentions)

        return outputs

    def forward_batch(self, batch: CTBatch, need_weights=False) -> \
            CodeTransformerOutput:
        output = self.forward(batch.tokens, input_node_types=batch.node_types, input_token_types=batch.token_types,
                              relative_distances=batch.relative_distances, attention_mask=batch.perm_mask,
                              pad_mask=1 - batch.pad_mask, target_mapping=batch.target_mapping, labels=batch.labels,
                              need_weights=need_weights)
        return output
