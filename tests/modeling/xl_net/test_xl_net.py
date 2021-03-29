import unittest

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_transformer.configuration.attention import AttentionType
from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling.constants import SOS_TOKEN
from code_transformer.modeling.xl_net.decoder import XLNetTransformerDecoder
from code_transformer.modeling.xl_net.xl_net_language_model import XLNetLMEncoder
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetNoPunctuation
from code_transformer.env import DATA_PATH_STAGE_2


class TestXLNet(unittest.TestCase):

    def _create_xl_net_model(self, data_manager, use_pointer_network=True, use_query_self_attention=False,
                             output_subtokens_per_token=6, num_languages=None):
        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()
        config = data_manager.load_config()

        transformer_config = dict(
            d_model=16,
            n_head=8,
            d_inner=32,
            ff_activation="gelu",
            dropout=0.2,
            n_layer=3,
            vocab_size=len(word_vocab),
            mem_len=1024
        )
        encoder_config = dict(
            vocab_size=len(word_vocab),
            num_token_types=len(token_type_vocab),
            num_languages=num_languages
        )
        decoder_config = dict(
            sos_id=config['preprocessing']['special_symbols'][SOS_TOKEN],
            n_layers=2,
            use_teacher_forcing=True,
            output_subtokens_per_token=output_subtokens_per_token,
            decoder_nhead=8,
            decoder_dim_feedforward=32,
            use_pointer_network=use_pointer_network,
            use_pointer_query_self_attention=use_query_self_attention,
            pointer_attention_type=AttentionType.MULTIHEAD
        )

        def init_model():
            encoder_config['transformer'] = transformer_config
            decoder_config['lm_encoder'] = XLNetLMEncoder(TransformerLMEncoderConfig(**encoder_config))
            model = XLNetTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

            num_params = sum([len(params.view(-1)) for params in model.parameters()])
            print(f"Model has {num_params} parameters")

            return model

        model = init_model()
        return model

    def test_xl_net(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='java-small', partition='valid')
        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager, num_sub_tokens_output=6,
                                                          use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)
        batch = next(iterator)

        model = self._create_xl_net_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=8e-3)

        tq = tqdm(range(100))
        for _ in tq:
            output = model.forward_batch(batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=output.loss.item())

        self.assertLess(output.loss.item(), 1)

    def test_xl_net_multilanguage(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='python,javascript,go,ruby', partition='train', infinite_loading=True)

        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager, num_sub_tokens_output=6,
                                                          use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)
        batch = next(iterator)

        model = self._create_xl_net_model(data_manager, num_languages=4)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        tq = tqdm(range(100))
        for i in tq:
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss, 1.5)