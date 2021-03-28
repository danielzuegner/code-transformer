from abc import ABC

from torch.nn import CrossEntropyLoss

from code_transformer.configuration.great_transformer import GreatTransformerConfig, GreatEncoderConfig
from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.modeling.constants import SOS_TOKEN, UNKNOWN_TOKEN
from code_transformer.modeling.great_transformer.transformer import GreatEncoderTransformerAdapter, \
    GreatTransformerDecoder
from code_transformer.modeling.modelmanager import GreatModelManager
from code_transformer.utils.loss import LabelSmoothingLoss


class GreatTransformerDecoderMixin(ExperimentSetup, ABC):

    @ex.capture(prefix="model")
    def _init_model(self, lm_encoder: dict, lm_decoder: dict, with_cuda: bool, label_smoothing=None):

        transformer_config = GreatTransformerConfig(**lm_encoder['transformer_config'])

        config = GreatEncoderConfig(**lm_encoder)

        num_edge_types = 0
        for d in self.relative_distances:
            if d in ["ancestor_sp", "sibling_sp"]:
                num_edge_types += 2
            elif d == "shortest_paths":
                num_edge_types += 1
        transformer_config.bias_dim = num_edge_types
        config.transformer_config = transformer_config

        if hasattr(self, 'word_vocab'):
            config.vocab_size = len(self.word_vocab.vocabulary)
        if hasattr(self, 'node_type_vocab'):
            config.num_node_types = len(self.node_type_vocab.vocabulary)
        if hasattr(self, "num_sub_tokens"):
            config.subtokens_per_token = self.num_sub_tokens
        if hasattr(self, 'num_languages'):
            config.num_languages = self.num_languages

        great_lm_encoder = GreatEncoderTransformerAdapter(config)

        if label_smoothing is None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
        else:
            loss_fct = LabelSmoothingLoss(label_smoothing)

        model_config = TransformerLMDecoderConfig(great_lm_encoder, sos_id=self.word_vocab[SOS_TOKEN],
                                                       unk_id=self.word_vocab[UNKNOWN_TOKEN], loss_fct=loss_fct,
                                                       output_subtokens_per_token=self.dataset_train.num_sub_tokens_output,
                                                       use_pointer_network=self.use_pointer_network if hasattr(self,
                                                                                                               "use_pointer_network") else False,
                                                       **lm_decoder)
        self.model_manager = GreatModelManager()
        self.model_lm = GreatTransformerDecoder(model_config)

        self.with_cuda = with_cuda
