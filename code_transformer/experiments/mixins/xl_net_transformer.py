from abc import ABC

from torch.nn import CrossEntropyLoss

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.experiments.experiment import ex
from code_transformer.experiments.xl_net.base import XLNetExperimentSetup
from code_transformer.modeling.constants import SOS_TOKEN, UNKNOWN_TOKEN
from code_transformer.modeling.modelmanager import XLNetModelManager
from code_transformer.modeling.xl_net.xl_net_language_model import XLNetLMEncoder, XLNetLanguageModel
from code_transformer.modeling.xl_net.decoder import XLNetTransformerDecoder
from code_transformer.utils.loss import LabelSmoothingLoss


class XLNetTransformerMixin(XLNetExperimentSetup, ABC):

    @ex.capture(prefix="model")
    def _init_model(self, lm_encoder: dict, lm_decoder: dict, with_cuda: bool, label_smoothing=None):

        if hasattr(self.dataset_train, 'num_sub_tokens_output'):
            num_sub_tokens_output = self.dataset_train.num_sub_tokens_output
        else:
            num_sub_tokens_output = 5

        config = TransformerLMEncoderConfig(**lm_encoder)

        if self.use_pretrained_model:
            loaded_config = self.pretrained_transformer_encoder_config
            if not config == self.pretrained_transformer_encoder_config:
                print(f"pretrained configuration differs from given configuration. Pretrained: "
                      f"{self.pretrained_transformer_encoder_config}, Given: {config}. Try merging...")
                loaded_config.input_nonlinearity = config.input_nonlinearity
                loaded_config.transformer['dropout'] = config.transformer['dropout']
            config = loaded_config

        transformer_config = dict(config.transformer)

        if hasattr(self, "word_vocab"):
            config.vocab_size = len(self.word_vocab)
        if hasattr(self, "token_type_vocab"):
            config.num_token_types = len(self.token_type_vocab)
        if hasattr(self, "node_type_vocab"):
            config.num_node_types = len(self.node_type_vocab)

        config.transformer = transformer_config

        xl_net_lm_encoder = XLNetLMEncoder(config)
        if self.use_pretrained_model:
            # Pretraining is done with language model. Thus, the encoder needs to be wrapped inside a Language model
            # first in order to be able to load the pretrained parameters

            xl_net_lm = XLNetLanguageModel(xl_net_lm_encoder)
            xl_net_lm.load_state_dict(self.pretrained_model_params)

        if label_smoothing is None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
        else:
            loss_fct = LabelSmoothingLoss(label_smoothing)

        model_config = TransformerLMDecoderConfig(xl_net_lm_encoder, sos_id=self.word_vocab[SOS_TOKEN],
                                                       unk_id=self.word_vocab[UNKNOWN_TOKEN], loss_fct=loss_fct,
                                                       output_subtokens_per_token=self.dataset_train.num_sub_tokens_output,
                                                       use_pointer_network=self.use_pointer_network if hasattr(self,
                                                                                                               "use_pointer_network") else False,
                                                       **lm_decoder)
        self.model_manager = XLNetModelManager()
        self.model_lm = XLNetTransformerDecoder(model_config)

        self.with_cuda = with_cuda
