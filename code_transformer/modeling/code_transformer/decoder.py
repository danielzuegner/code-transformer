from torch import nn

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.decoder.transformer import TransformerLMDecoder
from code_transformer.modeling.code_transformer.lm import TransformerLMEncoder
from code_transformer.preprocessing.datamanager.base import CTBatch


class CodeTransformerDecoder(TransformerLMDecoder):

    def __init__(self, config: TransformerLMDecoderConfig):
        if not isinstance(config.lm_encoder, nn.Module):
            config.lm_encoder = TransformerLMEncoder(
                TransformerLMEncoderConfig(**config.lm_encoder))

        super(CodeTransformerDecoder, self).__init__(config)

    def forward_batch(self, batch: CTBatch, need_weights=False):
        return self.forward(input_tokens=batch.tokens, input_node_types=batch.node_types,
                            input_token_types=batch.token_types,
                            relative_distances=batch.relative_distances, attention_mask=batch.perm_mask,
                            pad_mask=1 - batch.pad_mask, target_mapping=batch.target_mapping,
                            labels=batch.labels,
                            need_weights=need_weights,
                            extended_vocabulary_ids=batch.extended_vocabulary_ids,
                            pointer_pad_mask=batch.pointer_pad_mask,
                            languages=batch.languages)
