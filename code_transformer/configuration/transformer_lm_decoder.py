from torch import nn

from code_transformer.configuration.configuration_utils import ModelConfiguration
from code_transformer.configuration.attention import AttentionType


class TransformerLMDecoderConfig(ModelConfiguration):

    def __init__(self,
                 lm_encoder,  #: Union[TransformerLMEncoder, TransformerLMEncoderConfig],
                 sos_id: int,
                 unk_id=0,
                 n_layers=1,
                 decoder_dropout=0,
                 decoder_nhead=8,
                 decoder_dim_feedforward=2048,
                 decoder_activation="gelu",
                 use_teacher_forcing=False,
                 output_subtokens_per_token=5,
                 output_nonlinearity=None,
                 loss_fct=nn.CrossEntropyLoss(ignore_index=-1),
                 use_pointer_network=False,
                 use_pointer_query_linear=False,
                 use_pointer_query_self_attention=False,
                 concat_query_and_pointer=True,
                 attend_cls_token=False,
                 pointer_attention_type=AttentionType.MULTIHEAD,
                 target_vocab_size: int = None):
        r"""
        :param lm_encoder: The encoder on which the decoder should be built
        :param sos_id: The ID of the SOS token in the underlying vocabulary. Initially the decoder sequence will be
            populated with a single SOS token to have an input for the LSTM decoder
        :param n_layers: How many layers the decoder should have.
        :param decoder_dropout: Whether dropout should be applied in the decoder.
        :param use_teacher_forcing: If set, the previous label will be fed into the decoder instead of the
            previous prediction during training. Usually speeds up training but also introduces a gap between
            training and evaluation.
        :param target_vocab_size: If given, the model assumes separate vocabularies for input tokens and label tokens.
            The value specified here indicates the output distribution domain that is to be predicted
        """
        super(TransformerLMDecoderConfig, self).__init__()
        self.lm_encoder = lm_encoder
        self.sos_id = sos_id
        self.unk_id = unk_id
        self.n_layers = n_layers
        self.decoder_dropout = decoder_dropout
        self.decoder_nhead = decoder_nhead
        self.decoder_dim_feedforward = decoder_dim_feedforward
        self.decoder_activation = decoder_activation
        self.use_teacher_forcing = use_teacher_forcing
        self.output_subtokens_per_token = output_subtokens_per_token
        self.output_nonlinearity = output_nonlinearity
        self.loss_fct = loss_fct
        self.use_pointer_network = use_pointer_network
        self.pointer_attention_type = pointer_attention_type if isinstance(pointer_attention_type,
                                                                           AttentionType) else AttentionType(
            pointer_attention_type)
        self.use_pointer_query_linear = use_pointer_query_linear
        self.use_pointer_query_self_attention = use_pointer_query_self_attention
        self.concat_query_and_pointer = concat_query_and_pointer
        self.attend_cls_token = attend_cls_token
        assert not (
                    use_pointer_query_self_attention and use_pointer_query_linear), "Cannot set both query linear and query self attention"
        self.target_vocab_size = target_vocab_size
