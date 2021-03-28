from code_transformer.configuration.configuration_utils import ModelConfiguration


class TransformerLMEncoderConfig(ModelConfiguration):

    def __init__(self,
                 transformer,  #: Union[CodeTransformer, CodeTransformerCoreConfig],
                 vocab_size=32000,
                 num_node_types=None,
                 num_token_types=None,
                 subtokens_per_token=5,
                 input_nonlinearity=None,
                 num_languages=None):
        super(TransformerLMEncoderConfig, self).__init__()

        self.transformer = transformer
        self.vocab_size = vocab_size
        self.num_token_types = num_token_types
        self.num_node_types = num_node_types
        self.subtokens_per_token = subtokens_per_token
        self.input_nonlinearity = input_nonlinearity
        self.num_languages = num_languages
