import torch

from code_transformer.configuration.great_transformer import GreatEncoderConfig
from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.decoder.transformer import TransformerLMDecoder
from code_transformer.modeling.great_transformer.great_transformer import GreatEncoder
from code_transformer.modeling.code_transformer.code_transformer import TransformerOutput
from code_transformer.preprocessing.dataset.code_summarization import GreatBatch
from torch import nn


class GreatEncoderTransformerAdapter(GreatEncoder):
    def forward(self, **model_input):
        output = super(GreatEncoderTransformerAdapter, self).forward(**model_input)
        return TransformerOutput(output[:, 0, :].unsqueeze(1), None,
                                 [(output.transpose(0, 1), torch.zeros((1, output.shape[0], output.shape[2]), device=output.device))])


class GreatTransformerDecoder(TransformerLMDecoder):

    def __init__(self, config: TransformerLMDecoderConfig):
        if not isinstance(config.lm_encoder, nn.Module):
            config.transformer_lm_encoder = GreatEncoder(GreatEncoderConfig(**config.lm_encoder))

        config.lm_encoder.d_model = config.lm_encoder.transformer.hidden_dim

        super(GreatTransformerDecoder, self).__init__(config)

    def forward_batch(self, batch: GreatBatch):
        return self.forward(input_tokens=batch.tokens,
                            edge_ixs=batch.edge_ixs,
                            attention_mask=batch.attention_mask,
                            pad_mask=1 - batch.pad_mask,
                            labels=batch.labels,
                            pointer_pad_mask=batch.pointer_pad_mask,
                            extended_vocabulary_ids=batch.extended_vocabulary_ids,
                            languages=batch.languages)
