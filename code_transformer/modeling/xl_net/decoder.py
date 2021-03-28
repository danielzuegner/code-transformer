from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.decoder.transformer import TransformerLMDecoder
from torch import nn

from code_transformer.modeling.xl_net.xl_net_language_model import XLNetLMEncoder
from code_transformer.preprocessing.datamanager.base import CTBatch


class XLNetTransformerDecoder(TransformerLMDecoder):
    def __init__(self, config: TransformerLMDecoderConfig):
        if not isinstance(config.lm_encoder, nn.Module):
            config.lm_encoder = XLNetLMEncoder(TransformerLMEncoderConfig(**config.lm_encoder))

        super(XLNetTransformerDecoder, self).__init__(config)

    def forward_batch(self, batch: CTBatch):
        return self.forward(input_ids=batch.tokens,
                            token_type_ids=batch.token_types,
                            pad_mask=1 - batch.pad_mask,
                            attention_mask=batch.perm_mask,
                            target_mapping=batch.target_mapping,
                            labels=batch.labels,
                            extended_vocabulary_ids=batch.extended_vocabulary_ids,
                            pointer_pad_mask=batch.pointer_pad_mask,
                            languages=batch.languages)
