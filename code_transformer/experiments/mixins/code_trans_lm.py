from abc import ABC

from torch.nn import CrossEntropyLoss

from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.modeling.code_transformer.lm import TransformerLanguageModel
from code_transformer.modeling.modelmanager import CodeTransformerLMModelManager
from code_transformer.utils.loss import LabelSmoothingLoss


class CodeTransformerLanguageModelMixin(ExperimentSetup, ABC):

    @ex.capture(prefix="model")
    def _init_model(self, transformer_lm_encoder: dict, with_cuda: bool, output_nonlinearity=None,
                    label_smoothing=None):
        config = self.generate_transformer_lm_encoder_config(transformer_lm_encoder)

        if label_smoothing is None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
        else:
            loss_fct = LabelSmoothingLoss(label_smoothing)

        self.model_lm = TransformerLanguageModel(config, output_nonlinearity=output_nonlinearity, loss_fct=loss_fct)
        self.model_manager = CodeTransformerLMModelManager()
        self.model_config = config
        if self.use_pretrained_model:
            self.model_lm.load_state_dict(self.pretrained_model_params)

        self.with_cuda = with_cuda
