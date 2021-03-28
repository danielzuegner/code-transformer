from abc import ABC

from torch.nn import CrossEntropyLoss

from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.experiments.experiment import ex
from code_transformer.experiments.xl_net.base import XLNetExperimentSetup
from code_transformer.modeling.modelmanager.xl_net import XLNetLMModelManager
from code_transformer.modeling.xl_net.xl_net_language_model import XLNetLMEncoder, XLNetLanguageModel
from code_transformer.utils.loss import LabelSmoothingLoss


class XLNetLanguageModelingMixin(XLNetExperimentSetup, ABC):

    @ex.capture(prefix="model")
    def _init_model(self, transformer_lm_encoder: dict, with_cuda: bool, output_nonlinearity, label_smoothing=None):
        config = TransformerLMEncoderConfig(**transformer_lm_encoder)

        if hasattr(self, "word_vocab"):
            config.vocab_size = len(self.word_vocab)
        if hasattr(self, "token_type_vocab"):
            config.num_token_types = len(self.token_type_vocab)
        if hasattr(self, "node_type_vocab"):
            config.num_node_types = len(self.node_type_vocab)

        xl_net_lm_encoder = XLNetLMEncoder(config)

        if label_smoothing is None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
        else:
            loss_fct = LabelSmoothingLoss(label_smoothing)

        if hasattr(self.dataset_train, 'num_sub_tokens_output'):
            num_sub_tokens_output = self.dataset_train.num_sub_tokens_output
        else:
            num_sub_tokens_output = 5

        self.model_manager = XLNetLMModelManager()
        self.model_lm = XLNetLanguageModel(xl_net_lm_encoder, output_nonlinearity=output_nonlinearity,
                                           loss_fct=loss_fct, output_sub_tokens_per_token=num_sub_tokens_output)

        self.with_cuda = with_cuda
