import unittest

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_transformer.configuration.attention import AttentionType
from code_transformer.configuration.great_transformer import GreatEncoderConfig, GreatTransformerConfig
from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.modeling.constants import SOS_TOKEN
from code_transformer.modeling.great_transformer.transformer import GreatEncoderTransformerAdapter, \
    GreatTransformerDecoder
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetEdgeTypes
from env import DATA_PATH_STAGE_2


class TestGreat(unittest.TestCase):

    def _create_great_model(self, data_manager, use_pointer_network=True, output_subtokens_per_token=6):
        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()
        config = data_manager.load_config()

        transformer_config = dict(
            embed_dim=32,
            num_layers=3,
            num_heads=8,
            ff_dim=64,
            dropout_rate=0
        )
        encoder_config = dict(
            vocab_size=len(word_vocab),
            num_node_types=len(node_type_vocab)
        )
        decoder_config = dict(
            sos_id=config['preprocessing']['special_symbols'][SOS_TOKEN],
            n_layers=2,
            use_teacher_forcing=True,
            output_subtokens_per_token=output_subtokens_per_token,
            decoder_nhead=8,
            decoder_dim_feedforward=32,
            use_pointer_network=use_pointer_network,
            pointer_attention_type=AttentionType.MULTIHEAD
        )

        def init_model():
            encoder_config['transformer_config'] = GreatTransformerConfig(**transformer_config)
            decoder_config['lm_encoder'] = GreatEncoderTransformerAdapter(GreatEncoderConfig(**encoder_config))
            model = GreatTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

            num_params = sum([len(params.view(-1)) for params in model.parameters()])
            print(f"Model has {num_params} parameters")

            return model

        model = init_model()
        return model

    def test_great(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='python,javascript,go,ruby', partition='train', infinite_loading=True)
        dataset = CTCodeSummarizationDatasetEdgeTypes(data_manager, num_sub_tokens_output=6, use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)

        model = self._create_great_model(data_manager, use_pointer_network=True)
        optimizer = optim.Adam(model.parameters(), lr=8e-3)

        tq = tqdm(range(100))
        batch = next(iterator)
        for _ in tq:
            output = model.forward_batch(batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=output.loss.item())

        self.assertLess(output.loss.item(), 1)
