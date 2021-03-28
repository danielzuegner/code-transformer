from code_transformer.configuration.configuration_utils import ModelConfiguration


class CodeTransformerLayerConfig(ModelConfiguration):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 dim_feedforward=1024,
                 activation="gelu",
                 dropout=0.1,

                 num_relative_distances=1,
                 use_token_distances=False,
                 use_edge_embeddings=False,
                 use_content_content=True,
                 use_content_pos=True,
                 use_pos_content=True,
                 use_pos_pos=True, ):
        super(CodeTransformerLayerConfig, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout
        self.num_relative_distances = num_relative_distances
        self.use_token_distances = use_token_distances
        self.use_edge_embeddings = use_edge_embeddings
        self.use_content_content = use_content_content
        self.use_content_pos = use_content_pos
        self.use_pos_content = use_pos_content
        self.use_pos_pos = use_pos_pos


class CodeTransformerCoreConfig(ModelConfiguration):
    def __init__(self,
                 encoder_layer: CodeTransformerLayerConfig,
                 num_layers: int,
                 positional_encoding=None,
                 norm=None
                 ):
        super(CodeTransformerCoreConfig, self).__init__()
        if isinstance(encoder_layer, CodeTransformerLayerConfig):
            self.encoder_layer = CodeTransformerLayerConfig(**encoder_layer)
        else:
            self.encoder_layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm
        self.positional_encoding = positional_encoding
