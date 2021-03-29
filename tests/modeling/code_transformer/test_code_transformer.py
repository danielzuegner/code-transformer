import unittest

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from code_transformer.configuration.attention import AttentionType
from code_transformer.configuration.code_transformer import CodeTransformerLayerConfig, CodeTransformerCoreConfig
from code_transformer.configuration.transformer_lm_decoder import TransformerLMDecoderConfig
from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.modeling import data_utils
from code_transformer.modeling.code_transformer.code_transformer import CodeTransformerLayer, CodeTransformer
from code_transformer.modeling.code_transformer.decoder import CodeTransformerDecoder
from code_transformer.modeling.code_transformer.lm import TransformerLanguageModel, TransformerLMEncoder
from code_transformer.modeling.constants import SOS_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN
from code_transformer.preprocessing.datamanager.base import CTBatch, batch_to_device, batch_filter_distances
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager, \
    CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDatasetNoPunctuation
from code_transformer.preprocessing.dataset.lm import CTLanguageModelingDataset, \
    CTLanguageModelingDatasetNoPunctuation
from code_transformer.preprocessing.nlp.vocab import batch_decode
from code_transformer.utils.io import load_zipped
from code_transformer.utils.metrics import topk_accuracy
from code_transformer.env import DATA_PATH_STAGE_2

RUN_TESTS_ON_GPU = False


class TestCodeTransformer(unittest.TestCase):

    @staticmethod
    def generate_transformer_default_config(**kwargs) -> CodeTransformerCoreConfig:
        layer_conf = CodeTransformerLayerConfig(
            d_model=64,
            nhead=8,
            dim_feedforward=256,
            activation="gelu",
            dropout=0.1,
            num_relative_distances=2,
            use_token_distances=False,
            use_content_content=True,
            use_content_pos=True,
            use_pos_content=True,
            use_pos_pos=True,
        )
        layer_conf.update(**kwargs)
        conf = CodeTransformerCoreConfig(
            encoder_layer=layer_conf,
            num_layers=7
        )

        return conf

    def generate_language_model_default_config(self,
                                               transformer_config: CodeTransformerCoreConfig = None) \
            -> TransformerLMDecoderConfig:
        if transformer_config is None:
            transformer_config = TestCodeTransformer.generate_transformer_default_config()
        encoder_conf = TransformerLMEncoderConfig(
            transformer_config,
            vocab_size=113,
            num_node_types=5,
            num_token_types=13,
            subtokens_per_token=5,
            input_nonlinearity="tanh")

        return TransformerLMDecoderConfig(
            encoder_conf,
            sos_id=-1,
            output_nonlinearity=None
        )

    def attention_sanity_check(self, attentions, lengths, target_mapping_per_token):
        for att_content, att_query in attentions:
            att_content = att_content.permute((0, 3, 1, 2))
            att_query = att_query.permute((0, 3, 1, 2))

            assert att_content.max() <= 1
            assert att_content.min() >= 0
            if att_query.max() > 1:
                print(att_query.max())
            assert att_query.max() <= 1, att_query.max()
            assert att_query.min() >= 0
            assert ((att_query > 0).long().sum(-1).max((-1))[0].max(-1)[
                        0] <= lengths).all(), "Tokens attended padded tokens"
            assert ((att_content > 0).long().sum(-1).max((-1))[0].max(-1)[
                        0] <= lengths).all(), "Tokens attended padded tokens"

            att_query_agg = att_query.mean(1)
            tgt_nnz = target_mapping_per_token.nonzero()
            tgt_nnz = torch.cat([tgt_nnz, tgt_nnz[:, 1][:, None]], dim=1)

            if not (att_query_agg[tuple(tgt_nnz.t())] == 0).all():
                print(att_query_agg[tuple(tgt_nnz.t())].max())
            assert (att_query_agg[tuple(
                tgt_nnz.t())] == 0).all(), "Query stream targets can attend themselves, leading to data leakage!"

    def setup_mini_dataset(self):
        BATCH_SIZE = 13
        NUM_PREDICT = 5

        vocab_path = f"{DATA_PATH_STAGE_2}/python/vocabularies.p.gzip"
        self.word_vocab, self.token_type_vocab, self.node_type_vocab = load_zipped(vocab_path)
        torch.manual_seed(102030)
        data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2, "python", "train")
        dataset = CTLanguageModelingDataset(data_manager, num_labels_per_sample=NUM_PREDICT)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)
        return dataloader

    def test_no_relative_distances(self):

        dataloader = self.setup_mini_dataset()
        config = CodeTransformerCoreConfig(
            encoder_layer=CodeTransformerLayerConfig(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                activation="gelu",
                num_relative_distances=0,
                use_token_distances=True,
                use_content_content=True,
                use_content_pos=True,
                use_pos_content=True,
                use_pos_pos=True
            ),
            num_layers=4
        )
        language_model_config = TransformerLMDecoderConfig(
            lm_encoder=TransformerLMEncoderConfig(
                transformer=config,
                vocab_size=len(self.word_vocab.vocabulary),
                num_node_types=len(self.node_type_vocab.vocabulary),
                num_token_types=len(self.token_type_vocab.vocabulary)
            ),
            sos_id=-1
        )
        transformer_lm = TransformerLanguageModel(
            transformer_lm_encoder=language_model_config['lm_encoder'],
            output_nonlinearity=language_model_config['output_nonlinearity'],
            loss_fct=language_model_config['loss_fct'])
        batch: CTBatch = next(iter(dataloader))

        with self.assertRaises(AssertionError):
            transformer_lm.forward_batch(batch)
        batch = batch_filter_distances(batch, [])
        transformer_lm.forward_batch(batch)

    def test_no_token_distances(self):

        dataloader = self.setup_mini_dataset()
        config = CodeTransformerCoreConfig(
            encoder_layer=CodeTransformerLayerConfig(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                activation="gelu",
                num_relative_distances=4,
                use_token_distances=False,
                use_content_content=True,
                use_content_pos=True,
                use_pos_content=True,
                use_pos_pos=True
            ),
            num_layers=4
        )
        language_model_config = TransformerLMDecoderConfig(
            lm_encoder=TransformerLMEncoderConfig(
                transformer=config,
                vocab_size=len(self.word_vocab.vocabulary),
                num_node_types=len(self.node_type_vocab.vocabulary),
                num_token_types=len(self.token_type_vocab.vocabulary)
            ),
            sos_id=-1
        )

        transformer_lm = TransformerLanguageModel(
            transformer_lm_encoder=language_model_config['lm_encoder'],
            output_nonlinearity=language_model_config['output_nonlinearity'],
            loss_fct=language_model_config['loss_fct'])
        batch: CTBatch = next(iter(dataloader))
        transformer_lm.forward_batch(batch)

    def test_fails_no_distances(self):
        dataloader = self.setup_mini_dataset()
        config = CodeTransformerCoreConfig(
            encoder_layer=CodeTransformerLayerConfig(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                activation="gelu",
                num_relative_distances=0,
                use_token_distances=False,
                use_content_content=True,
                use_content_pos=True,
                use_pos_content=True,
                use_pos_pos=True),
            num_layers=4,
        )
        language_model_config = TransformerLMDecoderConfig(
            lm_encoder=TransformerLMEncoderConfig(
                config,
                vocab_size=len(self.word_vocab.vocabulary),
                num_node_types=len(self.node_type_vocab.vocabulary),
                num_token_types=len(self.token_type_vocab.vocabulary)),
            sos_id=-1,
        )
        with self.assertRaises(Exception):
            transformer_lm = TransformerLanguageModel(
                transformer_lm_encoder=language_model_config['lm_encoder'],
                output_nonlinearity=language_model_config['output_nonlinearity'],
                loss_fct=language_model_config['loss_fct'])
            batch: CTBatch = next(iter(dataloader))
            transformer_lm.forward_batch(batch)

    def test_mini_dataset(self):

        def evaluate_predictions(logits, labels, loss=None):
            correct = logits.argmax(-1) == labels
            all_correct = correct.prod(-1)
            correct_tokens = all_correct.float().mean().cpu().item()
            ret = dict(correct_tokens=correct_tokens)
            if loss is not None:
                ret['loss'] = loss.detach().cpu().item()
            return ret

        BATCH_SIZE = 13
        NUM_PREDICT = 5

        dataloader = self.setup_mini_dataset()

        config = CodeTransformerCoreConfig(
            encoder_layer=CodeTransformerLayerConfig(
                d_model=16,
                nhead=8,
                dim_feedforward=32,
                activation="gelu",
                num_relative_distances=4,
                use_token_distances=True,
                use_content_content=True,
                use_content_pos=True,
                use_pos_content=True,
                use_pos_pos=True),
            num_layers=4,
        )

        language_model_config = TransformerLMDecoderConfig(
            lm_encoder=TransformerLMEncoderConfig(
                config,
                vocab_size=len(self.word_vocab.vocabulary),
                num_node_types=len(self.node_type_vocab.vocabulary),
                num_token_types=len(self.token_type_vocab.vocabulary)),
            sos_id=-1
        )
        transformer_lm = TransformerLanguageModel(
            transformer_lm_encoder=language_model_config['lm_encoder'],
            output_nonlinearity=language_model_config['output_nonlinearity'],
            loss_fct=language_model_config['loss_fct'])
        batch: CTBatch = next(iter(dataloader))

        cuda = torch.cuda.is_available() and RUN_TESTS_ON_GPU
        if cuda:
            transformer_lm = transformer_lm.cuda()

        opt = optim.Adam(transformer_lm.parameters(), lr=1e-4)
        tq = tqdm(range(500))

        if RUN_TESTS_ON_GPU:
            with self.assertRaises(RuntimeError):
                # CPU input on CUDA model should fail
                output = transformer_lm.forward_batch(batch)
            batch = batch_to_device(batch, "cuda")

        assert not (batch.labels == self.word_vocab['</s>']).any().item()
        for _ in tq:
            output = transformer_lm.forward_batch(batch)
            output.loss.backward()
            opt.step()
            opt.zero_grad()
            evaluation = evaluate_predictions(output.logits, batch.labels)
            acc = evaluation['correct_tokens']
            tq.set_postfix(loss=output.loss.cpu().item(), acc=acc)

            predicted_tokens = output.logits.argmax(-1)
            generated_text = batch_decode(self.word_vocab, predicted_tokens)
            generated_text2 = [
                " ".join([
                    "_".join([self.word_vocab.reverse_lookup(subtoken.item()) for subtoken in token])
                    for token in sample
                ])
                for sample in predicted_tokens
            ]
            assert list(generated_text) == generated_text2
        assert acc > 0.98

    def create_test_input(self, model):
        class DistanceBinning(object):

            def __init__(self, n_bins):
                self.n_bins = n_bins

            def __call__(self, sample):
                return_dict = sample
                distances_list = []
                for dist_name in sample['distances'].keys():
                    dist: torch.Tensor = sample['distances'][dist_name]
                    if dist.dtype in [torch.float32, torch.float16, torch.float64]:
                        resh = dist.reshape(-1).cpu().numpy()
                        possible_distances = torch.tensor(histedges_equalA(resh, self.n_bins - 1), dtype=torch.float32)
                        resh_tensor = torch.tensor(resh)
                        indices = (resh_tensor[:, None] > possible_distances[None, :]).sum(-1).reshape(dist.shape)
                        distances_list.append((indices, possible_distances, dist_name))

                    elif dist.dtype in [torch.long, torch.int32, torch.int16, torch.int8, torch.int, torch.bool]:
                        dist = dist.clamp_max(1000)
                        # min_dist = dist[dist < 1000].min()
                        max_dist = (dist[dist < 1000]).max()
                        possible_distances = torch.cat([torch.unique(dist), torch.tensor([1000])])
                        if len(possible_distances) > self.n_bins:
                            print(len(possible_distances))
                            raise ValueError("number of possible distances larger than number of bins!")
                        n_pad = self.n_bins - len(possible_distances)
                        if n_pad > 0:
                            possible_distances = torch.cat([possible_distances, max_dist * torch.ones(n_pad,
                                                                                                      dtype=torch.long)])
                        indices = dist.clone().to(torch.long)
                        nnz = ((indices.unsqueeze(-1) == possible_distances[None, None, :].long())).nonzero()
                        indices[(nnz[:, 0], nnz[:, 1])] = nnz[:, 2].long()
                        indices[indices >= 1000] = max_dist + 1
                        distances_list.append((indices, possible_distances, dist_name))
                return_dict['distances'] = distances_list
                return return_dict

        def histedges_equalA(x, nbin):
            """
            https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area
            """
            pow = 0.5
            dx = np.diff(np.sort(x))
            tmp = np.cumsum(dx ** pow)
            tmp = np.pad(tmp, (1, 0), 'constant')
            return np.interp(np.linspace(0, tmp.max(), nbin + 1),
                             tmp,
                             np.sort(x))

        output_subtokens_per_token = model.output_subtokens_per_token
        batch_size = 7
        sequence_length = 32

        CLS_ID = 0
        SOS_ID = model.sos_id
        FUNC_NAME_IDX = 3
        vocab_size = model.transformer_lm_encoder.vocab_size
        num_node_types = model.transformer_lm_encoder.num_node_types
        num_token_types = model.transformer_lm_encoder.num_token_types
        subtokens_per_token = model.transformer_lm_encoder.subtokens_per_token

        lengths = torch.linspace(9, 27, steps=batch_size).to(torch.long)
        pad_mask = data_utils.pad_mask(lengths, max_len=sequence_length)

        test_token_sequence = torch.randint(low=5, high=vocab_size,
                                            size=[batch_size, sequence_length, subtokens_per_token])
        test_token_sequence[:, 0] = CLS_ID
        test_node_type_sequence = torch.randint(low=0, high=num_node_types,
                                                size=[batch_size, sequence_length])

        test_token_type_sequence = torch.randint(low=0, high=num_token_types,
                                                 size=[batch_size, sequence_length])

        target_mapping = torch.zeros((batch_size, 1, sequence_length))
        target_mapping[:, :, 0] = 1

        labels = test_token_sequence[:, FUNC_NAME_IDX, :].unsqueeze(1)
        fill_labels = torch.randint(low=5, high=vocab_size,
                                    size=[batch_size, 1,
                                          output_subtokens_per_token - subtokens_per_token])
        labels = torch.cat([labels, fill_labels], -1)

        sequence_dists = torch.arange(sequence_length)[:, None] - torch.arange(sequence_length)[None, :]
        batch = [dict(
            distances=dict(seq=sequence_dists, )
        )
            for _ in range(batch_size)]
        db = DistanceBinning(n_bins=64)
        ret = [db(b) for b in batch]
        distances = torch.stack([x['distances'][0][0] for x in ret])
        bins = torch.stack([x['distances'][0][1].t() for x in ret]).t()

        perm_mask = torch.zeros((batch_size, sequence_length, sequence_length))
        perm_mask[:, :, 0] = 1
        perm_mask[:, :, FUNC_NAME_IDX] = 1
        perm_mask[:, 0, 0] = 0
        perm_mask[:, FUNC_NAME_IDX, 0] = 0

        rel_distances = [(distances, bins), ]

        cuda = torch.cuda.is_available()
        if cuda:
            test_token_sequence = test_token_sequence.cuda()
            test_node_type_sequence = test_node_type_sequence.cuda()
            test_token_type_sequence = test_token_type_sequence.cuda()
            target_mapping = target_mapping.cuda()
            perm_mask = perm_mask.cuda()
            pad_mask = pad_mask.cuda()
            rel_distances = [(x[0].cuda(), x[1].cuda()) for x in rel_distances]
            labels = labels.cuda()

        return test_token_sequence, test_node_type_sequence, test_token_type_sequence, target_mapping, perm_mask, \
               pad_mask, rel_distances, labels

    # =========================================================================
    # Decoder tests
    # =========================================================================

    def run_test(self, model, n_steps=250, test_input=None):
        self.model = model

        if test_input is None:
            test_token_sequence, test_node_type_sequence, test_token_type_sequence, target_mapping, perm_mask, \
            pad_mask, rel_distances, labels = self.create_test_input(model)
        else:
            test_token_sequence, test_node_type_sequence, test_token_type_sequence, target_mapping, perm_mask, \
            pad_mask, rel_distances, labels = test_input

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        tq = tqdm(range(n_steps))
        accuracy = 0
        for it in tq:
            output = self.model.forward(input_tokens=test_token_sequence,
                                        input_node_types=test_node_type_sequence,
                                        relative_distances=rel_distances,
                                        input_token_types=test_token_type_sequence,
                                        attention_mask=perm_mask,
                                        pad_mask=1 - pad_mask,
                                        target_mapping=target_mapping,
                                        labels=labels,
                                        need_weights=True)
            loss = output[0]
            loss.backward()
            opt.step()
            opt.zero_grad()
            logits = output[1]
            accuracy = (logits.argmax(-1) == labels).float().mean().detach().cpu().numpy()
            tq.set_postfix(loss=loss.detach().cpu(), accuracy=accuracy)
            attentions = output[2]
            # self.attention_sanity_check(attentions, lengths, target_mapping_per_token)
        return accuracy

    def generate_transformer_config(self):
        conf = CodeTransformerCoreConfig(
            encoder_layer=CodeTransformerLayerConfig(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                activation="gelu",
                dropout=0.1,
                num_relative_distances=1,
                use_token_distances=False,
                use_content_content=True,
                use_content_pos=True,
                use_pos_content=True,
                use_pos_pos=True),
            num_layers=7)
        return conf

    def generate_lm_encoder_config(self):
        return TransformerLMEncoderConfig(
            transformer=self.generate_transformer_config(),
            vocab_size=113,
            num_node_types=5,
            num_token_types=13,
            subtokens_per_token=5,
            input_nonlinearity="tanh")

    def test_sample_targets(self):
        num_predict = 2
        max_seq_len = 7
        batch_size = 2

        lengths = torch.linspace(num_predict, max_seq_len - 1, steps=batch_size).to(torch.long)
        pad_mask = data_utils.pad_mask(lengths, max_len=max_seq_len)

        targets = data_utils.sample_targets(num_predict=num_predict,
                                            seq_len=max_seq_len,
                                            batch_size=batch_size, pad_mask=pad_mask)


    # =========================================================================
    # Pointer Network Tests
    # =========================================================================

    def _create_pointer_network_model(self, data_manager, use_pointer_network=True,
                                      attention_type=AttentionType.MULTIHEAD):
        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()
        config = data_manager.load_config()

        layer_config = dict(
            d_model=16,
            nhead=8,
            dim_feedforward=32,
            activation="gelu",
            num_relative_distances=4,
            use_content_content=True,
            use_content_pos=True,
            use_pos_content=True,
            use_pos_pos=True,
            use_token_distances=True,
            dropout=0.2
        )
        transformer_config = dict(
            num_layers=1
        )
        encoder_config = dict(
            vocab_size=len(word_vocab),
            num_token_types=len(token_type_vocab),
            num_node_types=len(node_type_vocab),
            input_nonlinearity='tanh'
        )
        decoder_config = dict(
            sos_id=config['preprocessing']['special_symbols'][SOS_TOKEN],
            n_layers=1,
            use_teacher_forcing=True,
            output_subtokens_per_token=6,
            use_pointer_network=use_pointer_network,
            pointer_attention_type=attention_type
        )

        def init_model():
            transformer_config['encoder_layer'] = CodeTransformerLayer(**layer_config)
            encoder_config['transformer'] = CodeTransformer(CodeTransformerCoreConfig(**transformer_config))
            decoder_config['lm_encoder'] = TransformerLMEncoder(
                TransformerLMEncoderConfig(**encoder_config))
            model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

            num_params = sum([len(params.view(-1)) for params in model.parameters()])
            print(f"Model has {num_params} parameters")

            return model

        model = init_model()
        return model

    def test_pointer_network(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='java-small', partition='valid')
        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()

        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager, num_sub_tokens_output=6,
                                                          use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)
        batch = next(iterator)

        len_vocab = len(word_vocab)
        # Artificially set an out of vocabulary token
        batch.tokens[0][1][0] = word_vocab[UNKNOWN_TOKEN]

        model = self._create_pointer_network_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Deterministic in-vocabulary label that is NOT part of the input => model uses Decoder to generate label
        tq = tqdm(range(100))
        for i in tq:
            secret_label = len_vocab - 2
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss.item(), 0.5)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertGreater(out.pointer_gates.exp().min(), 0.95)
        model.eval()
        model.forward_batch(batch)
        self.assertLess(out.loss.item(), 0.5)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertGreater(out.pointer_gates.exp().min(), 0.95)

        model = self._create_pointer_network_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random out-of-vocabulary label that is part of the input => model uses Pointer
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.extended_vocabulary_ids[0][5] = secret_label
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss.item(), 0.01)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertLess(out.pointer_gates.exp().max(), 0.05)
        # Pointer network should point to artificial position in input
        self.assertGreater(out.pointer_attentions[:, 0, batch.labels[0, 0, 0].item()].exp().min(), 0.95)

        model = self._create_pointer_network_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random label that is NOT part of the input => no chance
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertGreater(out.loss.item(), 1)

        model = self._create_pointer_network_model(data_manager, use_pointer_network=False)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random in-vocabulary label that is part of the input, but no pointer network => no chance
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertGreater(out.loss.item(), 1)

    # =========================================================================
    # Transformer Decoder Tests
    # =========================================================================

    def _create_transformer_decoder_model(self, data_manager, use_pointer_network=True, use_query_self_attention=False,
                                          output_subtokens_per_token=6, num_languages=None):
        word_vocab, token_type_vocab, node_type_vocab = data_manager.load_vocabularies()
        config = data_manager.load_config()

        layer_config = dict(
            d_model=16,
            nhead=8,
            dim_feedforward=32,
            activation="gelu",
            num_relative_distances=4,
            use_content_content=True,
            use_content_pos=True,
            use_pos_content=True,
            use_pos_pos=True,
            use_token_distances=True,
            dropout=0.2
        )
        transformer_config = dict(
            num_layers=1
        )
        encoder_config = dict(
            vocab_size=len(word_vocab),
            num_token_types=len(token_type_vocab),
            num_node_types=len(node_type_vocab),
            input_nonlinearity='tanh',
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
            transformer_config['encoder_layer'] = CodeTransformerLayer(**layer_config)
            encoder_config['transformer'] = CodeTransformer(CodeTransformerCoreConfig(**transformer_config))
            decoder_config['lm_encoder'] = TransformerLMEncoder(
                TransformerLMEncoderConfig(**encoder_config))
            model = CodeTransformerDecoder(TransformerLMDecoderConfig(**decoder_config))

            num_params = sum([len(params.view(-1)) for params in model.parameters()])
            print(f"Model has {num_params} parameters")

            return model

        model = init_model()
        return model

    def test_transformer_decoder(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='java-small', partition='valid')
        word_vocab, _, _ = data_manager.load_vocabularies()

        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager, num_sub_tokens_output=6,
                                                          use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)
        batch = next(iterator)

        len_vocab = len(word_vocab)
        # Artificially set an out of vocabulary token
        batch.tokens[0][1][0] = word_vocab[UNKNOWN_TOKEN]

        model = self._create_transformer_decoder_model(data_manager, use_query_self_attention=True)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Pointer query self attention
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.extended_vocabulary_ids[0][5] = secret_label
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss.item(), 0.01)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertLess(out.pointer_gates.exp().max(), 0.05)
        # Pointer network should point to artificial position in input
        self.assertGreater(out.pointer_attentions[:, 0, batch.labels[0, 0, 0].item()].exp().min(), 0.95)

        model = self._create_transformer_decoder_model(data_manager, use_pointer_network=True)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Deterministic in-vocabulary label that is NOT part of the input => model uses Decoder to generate label
        tq = tqdm(range(100))
        for i in tq:
            secret_label = len_vocab - 2
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item(), top1=topk_accuracy(1, out.logits, batch.labels))

        self.assertLess(out.loss.item(), 0.5)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertGreater(out.pointer_gates.exp().min(), 0.95)
        model.eval()
        model.forward_batch(batch)
        self.assertLess(out.loss.item(), 0.5)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertGreater(out.pointer_gates.exp().min(), 0.95)

        model = self._create_transformer_decoder_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random out-of-vocabulary label that is part of the input => model uses Pointer
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.extended_vocabulary_ids[0][5] = secret_label
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss.item(), 0.01)
        self.assertGreater(topk_accuracy(1, out.logits, batch.labels), 0.95)
        self.assertLess(out.pointer_gates.exp().max(), 0.05)
        # Pointer network should point to artificial position in input
        self.assertGreater(out.pointer_attentions[:, 0, batch.labels[0, 0, 0].item()].exp().min(), 0.95)

        model = self._create_transformer_decoder_model(data_manager)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random label that is NOT part of the input => no chance
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertGreater(out.loss.item(), 1)

        model = self._create_transformer_decoder_model(data_manager, use_pointer_network=False)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        out = None
        # Random in-vocabulary label that is part of the input, but no pointer network => no chance
        tq = tqdm(range(100))
        for i in tq:
            secret_label = torch.randint(0, len_vocab, (1,))
            batch.labels[0] = torch.tensor([secret_label])
            batch.tokens[0][2:] = torch.randint(0, len_vocab, (batch.tokens.shape[1] - 2, 5))
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertGreater(out.loss.item(), 1)

    def test_transformer_decoder_language_modeling(self):
        n_predict = 2
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2, language='java-small', partition='valid')
        dataset = CTLanguageModelingDatasetNoPunctuation(data_manager, use_pointer_network=True,
                                                         num_labels_per_sample=n_predict)
        dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=2)
        word_vocab, _, _ = data_manager.load_vocabularies()

        batch = next(iter(dataloader))
        batch.labels[:] = word_vocab[PAD_TOKEN]

        model = self._create_transformer_decoder_model(data_manager, output_subtokens_per_token=5)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        # random labels that are always at same position in extended_vocabulary_ids => Pointer finds them
        tq = tqdm(range(100))
        for _ in tq:
            batch.labels[:, :, 0] = torch.randint(len(word_vocab), size=(2, n_predict))
            batch.tokens[:, [0, 1], 0] = batch.labels[:, :, 0]
            batch.extended_vocabulary_ids[:, list(range(n_predict))] = batch.labels[:, :, 0]
            output = model.forward_batch(batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=output.loss.item())

        self.assertTrue(not torch.isnan(output.logits).any())
        self.assertLess(output.loss.item(), 1)

        model = self._create_transformer_decoder_model(data_manager, output_subtokens_per_token=5)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        # random labels that are NOT part of input => no chance
        tq = tqdm(range(100))
        for _ in tq:
            batch.labels[:, :, 0] = torch.randint(len(word_vocab), size=(2, n_predict))
            batch.tokens[:, [0, 1], 0] = batch.labels[:, :, 0]
            output = model.forward_batch(batch)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=output.loss.item())

        self.assertTrue(not torch.isnan(output.logits).any())
        self.assertGreater(output.loss.item(), 0.5)

    def test_transformer_decoder_multilanguage(self):
        data_manager = CTPreprocessedDataManager(DATA_PATH_STAGE_2,
                                                 language='python,javascript,go,ruby', partition='train',
                                                 infinite_loading=True)

        dataset = CTCodeSummarizationDatasetNoPunctuation(data_manager, num_sub_tokens_output=6,
                                                          use_pointer_network=True)
        dataloader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)

        iterator = iter(dataloader)
        batch = next(iterator)

        model = self._create_transformer_decoder_model(data_manager, num_languages=4)
        optimizer = optim.Adam(model.parameters(), lr=5e-3)

        tq = tqdm(range(100))
        for i in tq:
            out = model.forward_batch(batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tq.set_postfix(loss=out.loss.item())

        self.assertLess(out.loss, 1)
