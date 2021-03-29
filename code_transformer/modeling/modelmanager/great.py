from code_transformer.configuration.great_transformer import GreatTransformerConfig, GreatEncoderConfig
from code_transformer.modeling.constants import SOS_TOKEN, NUM_SUB_TOKENS
from code_transformer.modeling.great_transformer.transformer import GreatTransformerDecoder, \
    GreatEncoderTransformerAdapter
from code_transformer.modeling.modelmanager import ModelManager, TransformerLMDecoderConfig, MultiheadAttention
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.env import MODELS_SAVE_PATH, DATA_PATH_STAGE_2


class GreatModelManager(ModelManager):
    def __init__(self):
        super(GreatModelManager, self).__init__(MODELS_SAVE_PATH, 'great_code_summarization', 'GT')

    def load_model(self, run_id, snapshot_iteration, gpu=True):
        model_params = self.load_parameters(run_id, snapshot_iteration, gpu=gpu)
        config = self.load_config(run_id)
        model_config = self._prepare_model_config(config)

        language = config['data_setup']['language']
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language)

        decoder_config = model_config['lm_decoder']

        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()

        transformer_encoder_config = model_config['lm_encoder']
        transformer_encoder_config['num_node_types'] = len(node_type_vocab)
        transformer_encoder_config['vocab_size'] = len(word_vocab)
        transformer_encoder_config['transformer_config'] = GreatTransformerConfig(
            **transformer_encoder_config['transformer_config'])
        num_edge_types = 0
        for d in config['data_transforms']['relative_distances']:
            if d in ["ancestor_sp", "sibling_sp"]:
                num_edge_types += 2
            elif d == "shortest_paths":
                num_edge_types += 1
        transformer_encoder_config['transformer_config'].bias_dim = num_edge_types
        if ',' in data_manager.language:
            transformer_encoder_config['num_languages'] = len(data_manager.language.split(','))

        great_lm_encoder = GreatEncoderTransformerAdapter(GreatEncoderConfig(**transformer_encoder_config))

        decoder_config['sos_id'] = word_vocab[SOS_TOKEN]
        if 'num_subtokens_output' in config['data_setup']:
            decoder_config['output_subtokens_per_token'] = config['data_setup']['num_subtokens_output']
        else:
            decoder_config['output_subtokens_per_token'] = NUM_SUB_TOKENS

        if 'use_pointer_network' in config['data_setup']:
            decoder_config['use_pointer_network'] = config['data_setup']['use_pointer_network']

        decoder_config['lm_encoder'] = great_lm_encoder
        decoder_config['loss_fct'] = model_config['loss_fct']

        model = GreatTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

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
                model = GreatTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))
                model.load_state_dict(model_params)

        return model
