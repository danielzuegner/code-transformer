from torch.nn import MultiheadAttention

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.code_transformer.decoder import CodeTransformerDecoder
from code_transformer.modeling.constants import SOS_TOKEN, NUM_SUB_TOKENS
from code_transformer.modeling.modelmanager.base import ModelManager
from code_transformer.modeling.xl_net.decoder import XLNetTransformerDecoder
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.env import DATA_PATH_STAGE_2, MODELS_SAVE_PATH


class XLNetLMModelManager(ModelManager):

    def __init__(self, models_save_path):
        super(XLNetLMModelManager, self).__init__(models_save_path, 'xl_net_lm', 'XL-LM')


class XLNetModelManager(ModelManager):

    def __init__(self):
        super(XLNetModelManager, self).__init__(MODELS_SAVE_PATH,
                                                'xl_net_code_summarization', 'XL')

    def load_model(self, run_id, snapshot_iteration, gpu=True):
        model_params = self.load_parameters(run_id, snapshot_iteration, gpu=gpu)
        config = self.load_config(run_id)
        model_config = self._prepare_model_config(config)

        language = config['data_setup']['language']
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language)

        decoder_config = model_config['lm_decoder']

        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()

        transformer_encoder_config = model_config['lm_encoder']
        transformer_encoder_config['num_token_types'] = len(token_type_vocab)
        transformer_encoder_config['vocab_size'] = len(word_vocab)

        decoder_config['sos_id'] = word_vocab[SOS_TOKEN]
        if 'num_subtokens_output' in config['data_setup']:
            decoder_config['output_subtokens_per_token'] = config['data_setup']['num_subtokens_output']
        else:
            decoder_config['output_subtokens_per_token'] = NUM_SUB_TOKENS

        if 'use_pointer_network' in config['data_setup']:
            decoder_config['use_pointer_network'] = config['data_setup']['use_pointer_network']

        decoder_config['lm_encoder'] = transformer_encoder_config
        decoder_config['loss_fct'] = model_config['loss_fct']

        model = XLNetTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

        try:
            model.load_state_dict(model_params)
        except RuntimeError:
            # In most cases, this is due to the legacy issue with encoder_self_attention
            model.add_module('encoder_self_attention',
                             MultiheadAttention(model.d_model, decoder_config['decoder_nhead'],
                                                dropout=decoder_config['decoder_dropout']))
            try:
                model.load_state_dict(model_params)
            except RuntimeError:
                decoder_config['concat_query_and_pointer'] = False
                model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))
                model.load_state_dict(model_params)

        return model
