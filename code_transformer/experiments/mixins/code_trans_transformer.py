from abc import ABC

from torch.nn import CrossEntropyLoss

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.experiments.experiment import ExperimentSetup, ex
from code_transformer.modeling.constants import SOS_TOKEN, UNKNOWN_TOKEN
from code_transformer.modeling.code_transformer.decoder import CodeTransformerDecoder
from code_transformer.modeling.modelmanager import CodeTransformerModelManager
from code_transformer.utils.loss import LabelSmoothingLoss


class CodeTransformerDecoderMixin(ExperimentSetup, ABC):

    @ex.capture(prefix="model")
    def _init_model(self, lm_encoder: dict, lm_decoder: dict, with_cuda: bool, label_smoothing=None):
        if hasattr(self.dataset_train, 'num_sub_tokens_output'):
            num_sub_tokens_output = self.dataset_train.num_sub_tokens_output
        else:
            num_sub_tokens_output = 5

        self.model_manager = CodeTransformerModelManager()
        if hasattr(self, 'pretrained_model'):
            self.model_lm = self.pretrained_model
            self.model_lm.output_subtokens_per_token = num_sub_tokens_output
        else:
            lm_encoder = self.generate_transformer_lm_encoder_config(lm_encoder)

            if label_smoothing is None:
                loss_fct = CrossEntropyLoss(ignore_index=-1)
            else:
                loss_fct = LabelSmoothingLoss(label_smoothing)

            model_config = TransformerLMDecoderConfig(
                lm_encoder=lm_encoder,
                sos_id=self.word_vocab[SOS_TOKEN],
                unk_id=self.word_vocab[UNKNOWN_TOKEN],
                loss_fct=loss_fct,
                use_pointer_network=self.use_pointer_network if hasattr(self, "use_pointer_network") else False,
                output_subtokens_per_token=num_sub_tokens_output,
                target_vocab_size=len(self.method_name_vocab) if self.use_separate_vocab else None,
                **lm_decoder
            )

            self.model_lm = CodeTransformerDecoder(model_config)

        if hasattr(self, "freeze_encoder_layers"):
            layers = self.model_lm.lm_encoder.transformer.layers
            freeze_encoder_layers = len(layers) if self.freeze_encoder_layers == 'all' else min(len(layers),
                                                                                                self.freeze_encoder_layers)
            print(f"Freezing {freeze_encoder_layers} encoder layers.")
            for i in range(freeze_encoder_layers):
                for param in layers[i].parameters():
                    param.requires_grad = False

        self.with_cuda = with_cuda
