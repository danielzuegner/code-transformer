from torch import nn
from torch.nn import MultiheadAttention

from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.code_transformer.decoder import CodeTransformerDecoder
from code_transformer.modeling.constants import SOS_TOKEN, NUM_SUB_TOKENS
from code_transformer.modeling.modelmanager.base import ModelManager
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.env import MODELS_SAVE_PATH, DATA_PATH_STAGE_2


class CodeTransformerModelManager(ModelManager):

    def __init__(self):
        super(CodeTransformerModelManager, self).__init__(MODELS_SAVE_PATH,
                                                          'ct_code_summarization',
                                                          'CT')

    def load_model(self, run_id, snapshot_iteration, gpu=True):
        model_params = self.load_parameters(run_id, snapshot_iteration, gpu=gpu)
        config = self.load_config(run_id)
        model_config = self._prepare_model_config(config)

        language = config['data_setup']['language']
        use_only_ast = config['data_setup']['use_only_ast'] if 'use_only_ast' in config['data_setup'] else False
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language)

        decoder_config = model_config['lm_decoder']

        vocabularies = data_manager.load_vocabularies()
        if len(vocabularies) == 3:
            word_vocab, token_type_vocab, node_type_vocab = vocabularies
            use_separate_vocab = False
        else:
            word_vocab, token_type_vocab, node_type_vocab, method_name_vocab = vocabularies
            use_separate_vocab = True

        encoder_config = model_config['lm_encoder']
        encoder_config['num_node_types'] = len(node_type_vocab)
        if use_only_ast:
            encoder_config['num_token_types'] = None
        else:
            encoder_config['num_token_types'] = len(token_type_vocab)
        encoder_config['vocab_size'] = len(word_vocab)
        encoder_config['transformer']['encoder_layer']['num_relative_distances'] = len(
            config['data_transforms']['relative_distances'])
        decoder_config['sos_id'] = word_vocab[SOS_TOKEN]
        if 'num_subtokens_output' in config['data_setup']:
            decoder_config['output_subtokens_per_token'] = config['data_setup']['num_subtokens_output']
        else:
            decoder_config['output_subtokens_per_token'] = NUM_SUB_TOKENS

        if 'use_pointer_network' in config['data_setup']:
            decoder_config['use_pointer_network'] = config['data_setup']['use_pointer_network']

        if ',' in data_manager.language:
            encoder_config['num_languages'] = len(data_manager.language.split(','))

        decoder_config['lm_encoder'] = encoder_config
        decoder_config['loss_fct'] = model_config['loss_fct']

        if use_separate_vocab:
            decoder_config['target_vocab_size'] = len(method_name_vocab)

        model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

        try:
            model.load_state_dict(model_params)
        except RuntimeError as e:
            # In most cases, this is due to the legacy issue with encoder_self_attention
            model.add_module('encoder_self_attention',
                             MultiheadAttention(model.d_model, decoder_config['decoder_nhead'],
                                                dropout=decoder_config['decoder_dropout']))
            try:
                model.load_state_dict(model_params)
            except RuntimeError:
                decoder_config['concat_query_and_pointer'] = False
                model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))
                try:
                    model.load_state_dict(model_params)
                except:
                    decoder_config['concat_query_and_pointer'] = True
                    model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))
                    model.lm_encoder.language_embedding = None
                    try:
                        model.load_state_dict(model_params)
                    except:
                        decoder_config['concat_query_and_pointer'] = False
                        model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

                        class PositionalEncodingMock(nn.Module):
                            def forward(self, x, position):
                                return x

                        model.positional_encoding = PositionalEncodingMock()
                        model.load_state_dict(model_params)

        return model


class CodeTransformerLMModelManager(CodeTransformerModelManager):
    def __init__(self):
        super(CodeTransformerLMModelManager, self).__init__()
        self.model_type = 'ct_lm'
        self.run_prefix = 'CT-LM'
