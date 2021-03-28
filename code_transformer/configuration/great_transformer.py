from code_transformer.configuration.configuration_utils import ModelConfiguration


class GreatTransformerConfig(ModelConfiguration):
    def __init__(self,
                 num_layers: int,
                 positional_encoding=None,
                 embed_dim=256,
                 num_heads=8,
                 ff_dim=1024,
                 dropout_rate=0.1,
                 is_encoder_decoder=False
                 ):
        super(GreatTransformerConfig, self).__init__()

        self.num_layers = num_layers
        self.positional_encoding = positional_encoding

        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim
        self.attention_dim = embed_dim
        self.bias_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.is_encoder_decoder = is_encoder_decoder


class GreatEncoderConfig(ModelConfiguration):

    def __init__(self,
                 transformer_config: GreatTransformerConfig,
                 vocab_size=32000,
                 num_node_types=None,
                 subtokens_per_token=5,
                 num_languages=None
                 ):
        super(GreatEncoderConfig, self).__init__()

        self.transformer_config = transformer_config
        self.vocab_size = vocab_size
        self.num_node_types = num_node_types
        self.subtokens_per_token = subtokens_per_token
        self.num_languages = num_languages
