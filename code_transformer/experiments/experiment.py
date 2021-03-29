import random
import signal
import sys
from abc import abstractmethod
from itertools import islice
from statistics import mean

import torch
from sacred import Experiment
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_transformer.configuration.transformer_lm_encoder import TransformerLMEncoderConfig
from code_transformer.experiments.log import ExperimentLogger, TensorboardLogger
from code_transformer.modeling.constants import PAD_TOKEN, UNKNOWN_TOKEN, EOS_TOKEN, NUM_SUB_TOKENS
from code_transformer.modeling.modelmanager import ModelManager
from code_transformer.modeling.modelmanager.code_transformer import CodeTransformerModelManager, \
    CodeTransformerLMModelManager
from code_transformer.preprocessing.datamanager.base import batch_filter_distances, batch_to_device, \
    DataLoaderWrapper, BufferedDataManager
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.dataset.lm import CTLanguageModelingDataset, \
    CTLanguageModelingDatasetNoPunctuation
from code_transformer.preprocessing.graph.binning import ExponentialBinning, EqualBinning
from code_transformer.preprocessing.graph.distances import DistanceBinning
from code_transformer.preprocessing.graph.transform import MaxDistanceMaskTransform, TokenDistancesTransform
from code_transformer.utils.metrics import top1_accuracy, topk_accuracy, precision, recall, f1_score, \
    non_trivial_words_accuracy, micro_f1_score, rouge_2, rouge_l
from code_transformer.utils.timing import Timing
from code_transformer.env import MODELS_SAVE_PATH, LOGS_PATH, DATA_PATH_STAGE_2

ex = Experiment(base_dir='../../', interactive=False)


class ExperimentSetup:

    def __init__(self):
        self._init_config()
        self._init_data_transforms()
        self._init_data()
        self._init_transfer_learning()
        self._init_model()
        self._init_optimizer()

    @ex.capture
    def _init_config(self, _config):
        self.config = _config

    @ex.capture(prefix="data_transforms")
    def _init_data_transforms(self, max_distance_mask, relative_distances, distance_binning):
        self.max_distance_mask = None if max_distance_mask is None else MaxDistanceMaskTransform(max_distance_mask)
        self.relative_distances = [] if relative_distances is None else relative_distances

        if distance_binning['type'] == 'exponential':
            trans_func = ExponentialBinning(distance_binning['growth_factor'])
        else:
            trans_func = EqualBinning()
        self.distance_binning = {
            'n_fixed_bins': distance_binning['n_fixed_bins'],
            'trans_func': trans_func
        }

    @ex.capture(prefix="data_setup")
    def _init_data(self, language, num_predict, use_validation=False, mini_dataset=False,
                   use_no_punctuation=False, use_pointer_network=False, sort_by_length=False, shuffle=True,
                   chunk_size=None, filter_language=None, dataset_imbalance=None, num_sub_tokens=NUM_SUB_TOKENS):
        self.data_manager = CTBufferedDataManager(DATA_PATH_STAGE_2, language, shuffle=shuffle,
                                                  infinite_loading=True,
                                                  mini_dataset=mini_dataset, size_load_buffer=10000,
                                                  sort_by_length=sort_by_length, chunk_size=chunk_size,
                                                  filter_language=filter_language, dataset_imbalance=dataset_imbalance)
        self.word_vocab, self.token_type_vocab, self.node_type_vocab = self.data_manager.load_vocabularies()

        token_distances = None
        if TokenDistancesTransform.name in self.relative_distances:
            num_bins = self.data_manager.load_config()['binning']['num_bins']
            token_distances = TokenDistancesTransform(
                DistanceBinning(num_bins, self.distance_binning['n_fixed_bins'], self.distance_binning['trans_func']))

        self.num_predict = num_predict
        self.use_pointer_network = use_pointer_network
        self.use_separate_vocab = False  # For language modeling we always only operate on the method body vocabulary

        if use_no_punctuation:
            self.dataset_train = CTLanguageModelingDatasetNoPunctuation(self.data_manager,
                                                                        token_distances=token_distances,
                                                                        max_distance_mask=self.max_distance_mask,
                                                                        num_labels_per_sample=num_predict,
                                                                        use_pointer_network=use_pointer_network,
                                                                        num_sub_tokens=num_sub_tokens)
        else:
            self.dataset_train = CTLanguageModelingDataset(self.data_manager, token_distances=token_distances,
                                                           max_distance_mask=self.max_distance_mask,
                                                           num_labels_per_sample=num_predict,
                                                           use_pointer_network=use_pointer_network,
                                                           num_sub_tokens=num_sub_tokens)

        self.use_validation = use_validation
        if self.use_validation:
            data_manager_validation = CTBufferedDataManager(DATA_PATH_STAGE_2, language, partition="valid",
                                                            shuffle=True, infinite_loading=True,
                                                            mini_dataset=mini_dataset, size_load_buffer=10000,
                                                            filter_language=filter_language,
                                                            dataset_imbalance=dataset_imbalance)
            if use_no_punctuation:
                self.dataset_validation = CTLanguageModelingDatasetNoPunctuation(data_manager_validation,
                                                                                 token_distances=token_distances,
                                                                                 max_distance_mask=self.max_distance_mask,
                                                                                 num_labels_per_sample=num_predict,
                                                                                 use_pointer_network=use_pointer_network,
                                                                                 num_sub_tokens=num_sub_tokens)
            else:
                self.dataset_validation = CTLanguageModelingDataset(data_manager_validation,
                                                                    token_distances=token_distances,
                                                                    max_distance_mask=self.max_distance_mask,
                                                                    num_labels_per_sample=num_predict,
                                                                    use_pointer_network=use_pointer_network,
                                                                    num_sub_tokens=num_sub_tokens)
        self.dataset_validation_creator = \
            lambda infinite_loading: self._create_validation_dataset(DATA_PATH_STAGE_2,
                                                                     language,
                                                                     use_no_punctuation,
                                                                     token_distances,
                                                                     infinite_loading,
                                                                     num_predict,
                                                                     use_pointer_network,
                                                                     filter_language,
                                                                     dataset_imbalance,
                                                                     num_sub_tokens)

    def _create_validation_dataset(self, data_location, language, use_no_punctuation, token_distances,
                                   infinite_loading, num_predict, use_pointer_network, filter_language,
                                   dataset_imbalance, num_sub_tokens):
        data_manager_validation = CTBufferedDataManager(data_location, language, partition="valid",
                                                        shuffle=True, infinite_loading=infinite_loading,
                                                        size_load_buffer=10000, filter_language=filter_language,
                                                        dataset_imbalance=dataset_imbalance)
        if use_no_punctuation:
            return CTLanguageModelingDatasetNoPunctuation(data_manager_validation,
                                                          token_distances=token_distances,
                                                          max_distance_mask=self.max_distance_mask,
                                                          num_labels_per_sample=num_predict,
                                                          use_pointer_network=use_pointer_network,
                                                          num_sub_tokens=num_sub_tokens)
        else:
            return CTLanguageModelingDataset(data_manager_validation,
                                             token_distances=token_distances,
                                             max_distance_mask=self.max_distance_mask,
                                             num_labels_per_sample=num_predict,
                                             use_pointer_network=use_pointer_network,
                                             num_sub_tokens=num_sub_tokens)

    @ex.capture(prefix="transfer_learning")
    def _init_transfer_learning(self, use_pretrained_model=False, model_type=None, run_id=None,
                                snapshot_iteration=None, cpu=False, freeze_encoder_layers=None):
        assert not use_pretrained_model or (
                run_id is not None
                and snapshot_iteration is not None
                and model_type is not None), "model_type, run_id and snapshot_iteration have to be provided if " \
                                             "use_pretrained_model is set"

        self.use_pretrained_model = use_pretrained_model
        if use_pretrained_model:

            print(
                f"Using Transfer Learning. Loading snapshot snapshot-{snapshot_iteration} from run {run_id} in collection "
                f"{model_type} ")

            if model_type == 'ct_code_summarization':
                model_manager = CodeTransformerModelManager()
                pretrained_model = model_manager.load_model(run_id, snapshot_iteration, gpu=not cpu)
                self.pretrained_model = pretrained_model
            elif model_type == 'ct_lm':
                model_manager = CodeTransformerLMModelManager()
                pretrained_model = model_manager.load_model(run_id, snapshot_iteration, gpu=not cpu)
                self.pretrained_model = pretrained_model
            else:
                model_manager = ModelManager(MODELS_SAVE_PATH, model_type)

                self.pretrained_model_params = model_manager.load_parameters(run_id, snapshot_iteration, gpu=not cpu)
                encoder_config = model_manager.load_config(run_id)['model']['transformer_lm_encoder']
                self.pretrained_transformer_encoder_config = TransformerLMEncoderConfig(**encoder_config)

            if freeze_encoder_layers is not None:
                self.freeze_encoder_layers = freeze_encoder_layers

    def generate_transformer_lm_encoder_config(self, transformer_lm_encoder: dict) -> TransformerLMEncoderConfig:
        config = TransformerLMEncoderConfig(**transformer_lm_encoder)
        if self.use_pretrained_model:
            loaded_config = self.pretrained_transformer_encoder_config
            if not config == self.pretrained_transformer_encoder_config:
                print(f"pretrained configuration differs from given configuration. Pretrained: "
                      f"{self.pretrained_transformer_encoder_config}, Given: {config}. Try merging...")
                loaded_config.input_nonlinearity = config.input_nonlinearity
                loaded_config.transformer['encoder_layer']['dropout'] = config.transformer['encoder_layer']['dropout']
                loaded_config.transformer['encoder_layer']['activation'] \
                    = config.transformer['encoder_layer']['activation']
            config = loaded_config

        transformer_config = dict(config.transformer)

        if hasattr(self, "word_vocab"):
            config.vocab_size = len(self.word_vocab)
        if hasattr(self, "token_type_vocab"):
            if hasattr(self, "use_only_ast") and self.use_only_ast:
                config.num_token_types = None
            else:
                config.num_token_types = len(self.token_type_vocab)
        if hasattr(self, "node_type_vocab"):
            config.num_node_types = len(self.node_type_vocab)
        if hasattr(self, "relative_distances"):
            encoder_layer_config = dict(transformer_config['encoder_layer'])
            encoder_layer_config['num_relative_distances'] = len(self.relative_distances)
            transformer_config['encoder_layer'] = encoder_layer_config
        if hasattr(self, "num_sub_tokens"):
            config.subtokens_per_token = self.num_sub_tokens
        if hasattr(self, 'num_languages'):
            config.num_languages = self.num_languages

        config.transformer = transformer_config
        return config

    @abstractmethod
    def _init_model(self, *args, **kwargs):
        self.model_lm = None
        self.with_cuda = True
        self.model_manager = None

    @ex.capture(prefix="optimizer")
    def _init_optimizer(self, learning_rate, reg_scale, scheduler=None, scheduler_params=None, optimizer="Adam"):
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_lm.parameters(), lr=learning_rate, weight_decay=reg_scale)
        elif optimizer == 'Momentum':
            self.optimizer = optim.SGD(self.model_lm.parameters(), lr=learning_rate, weight_decay=reg_scale,
                                       momentum=0.95, nesterov=True)

        self.scheduler = None
        if scheduler == 'OneCycleLR':
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, **scheduler_params)
        elif scheduler == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **scheduler_params)

    def _init_metrics(self, metrics):
        self.metrics = dict()
        pad_id = self.word_vocab[PAD_TOKEN]
        unk_id = self.word_vocab[UNKNOWN_TOKEN]
        for metric in metrics:
            if metric == 'top1_accuracy':
                self.metrics[metric] = top1_accuracy
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: top1_accuracy(logits, labels,
                                                                                        unk_id=unk_id, pad_id=pad_id)
            elif metric == 'top5_accuracy':
                self.metrics[metric] = lambda logits, labels: topk_accuracy(5, logits, labels)
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: topk_accuracy(5, logits, labels,
                                                                                        unk_id=unk_id, pad_id=pad_id)
            elif metric == 'precision':
                self.metrics[metric] = lambda logits, labels: precision(logits, labels, pad_id=pad_id)
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: precision(logits, labels, pad_id=pad_id,
                                                                                    unk_id=unk_id)
            elif metric == 'recall':
                self.metrics[metric] = lambda logits, labels: recall(logits, labels, pad_id=pad_id)
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: recall(logits, labels, pad_id=pad_id,
                                                                                 unk_id=unk_id)
            elif metric == 'f1_score':
                self.metrics[metric] = lambda logits, labels: f1_score(logits, labels, pad_id=pad_id)
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: f1_score(logits, labels, pad_id=pad_id,
                                                                                   unk_id=unk_id)
            elif metric == 'non_trivial_accuracy':
                self.metrics[metric] = lambda logits, labels: non_trivial_words_accuracy(logits, labels, pad_id)
                self.metrics[f"{metric}_no_unk"] = lambda logits, labels: non_trivial_words_accuracy(logits, labels,
                                                                                                     pad_id,
                                                                                                     unk_id=unk_id)
            elif metric == 'micro_f1_score':
                self.metrics[metric] = lambda logits, labels: micro_f1_score(logits, labels, pad_id=pad_id,
                                                                             unk_id=unk_id)
            elif metric == 'rouge_2':
                self.metrics[metric] = lambda logits, labels: rouge_2(logits, labels, pad_id=pad_id)
            elif metric == 'rouge_l':
                self.metrics[metric] = lambda logits, labels: rouge_l(logits, labels, pad_id=pad_id)

    @ex.capture(prefix="training")
    def train(self, batch_size, simulated_batch_size, random_seed, metrics,
              validate_every=None,
              persistent_snapshot_every=None, simulated_batch_size_valid=None, early_stopping_patience=10,
              max_validation_samples=10000, accumulate_tokens_batch=False):

        if self.with_cuda:
            self.model_lm = self.model_lm.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"

        run_id = self.model_manager.generate_run_name()

        self.logger = ExperimentLogger("experiment",
                                       TensorboardLogger(f"{LOGS_PATH}/{self.model_manager.model_type}/{run_id}"))
        self.logger.info(f"===============================================")
        self.logger.info(f"Starting run {run_id}")
        self.logger.info(f"===============================================")

        self.model_manager.save_config(run_id, self.config)
        early_stopping = EarlyStopping(self.model_manager, run_id, early_stopping_patience)

        num_params = sum([len(params.view(-1)) for params in self.model_lm.parameters()])
        self.logger.info(f"Start training model with {num_params} parameters")
        self.logger.info(f"Model setup: {self.model_lm}")

        self._init_metrics(metrics)

        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # Simulated batches
        simulated_batch_size = batch_size if simulated_batch_size is None else simulated_batch_size
        assert simulated_batch_size % batch_size == 0, "simulated_batch_size must be a multiple of batch_size"
        num_simulated_batches = simulated_batch_size // batch_size

        # Main train loop
        train_step = 0
        dataloader = DataLoader(self.dataset_train, batch_size=batch_size, collate_fn=self.dataset_train.collate_fn)

        if self.use_validation:
            if simulated_batch_size_valid is None:
                simulated_batch_size_valid = simulated_batch_size
            num_simulated_batches_valid = simulated_batch_size_valid // batch_size
            dataloader_validation = iter(DataLoader(self.dataset_validation, batch_size=batch_size,
                                                    collate_fn=self.dataset_validation.collate_fn))

        n_tokens_accumulate_batch = None
        if accumulate_tokens_batch:
            n_tokens_accumulate_batch = 0

        epoch = 1
        progress_bar = tqdm(total=int(self.data_manager.approximate_total_samples() / batch_size))
        progress_bar.set_description(f"Epoch {epoch}")

        # Ensure graceful shutdown when training is interrupted
        signal.signal(signal.SIGINT, self._handle_shutdown)

        with Timing() as t:
            for it, batch in enumerate(dataloader):
                self.logger.log_time(t.measure() / batch_size, "dataloader_seconds/sample",
                                     train_step * simulated_batch_size + (it % num_simulated_batches) * batch_size)
                # Calculate gradients
                batch = batch_filter_distances(batch, self.relative_distances)
                model_out = self._train_step(batch, num_simulated_batches)
                self.logger.log_time(t.measure() / batch_size, "model_seconds/sample",
                                     train_step * simulated_batch_size + (it % num_simulated_batches) * batch_size)

                # Log actual predicted words and labels
                self.logger.log_text("input/train",
                                     str([[self.word_vocab.reverse_lookup(st.item()) for st in token
                                           if st.item() != self.word_vocab[PAD_TOKEN]
                                           and st.item() != self.word_vocab[EOS_TOKEN]]
                                          for token in batch.tokens[0]]))
                self.logger.log_text("predicted words/train", str(self._decode_predicted_words(model_out, batch)))
                self.logger.log_text("labels/train", str(self._decode_labels(batch)))

                # Calculate metrics
                evaluation = self._evaluate_predictions(model_out.logits, batch.labels, loss=model_out.loss)
                self.logger.log_sub_batch_metrics(evaluation)

                if accumulate_tokens_batch:
                    n_tokens_accumulate_batch += batch.sequence_lengths.sum().item()

                # Gradient accumulation: only update gradients every num_simulated_batches step
                if not accumulate_tokens_batch and it % num_simulated_batches == (num_simulated_batches - 1) \
                        or accumulate_tokens_batch and n_tokens_accumulate_batch > simulated_batch_size:
                    if accumulate_tokens_batch:
                        n_tokens_accumulate_batch = 0
                    train_step += 1

                    total_norm = 0
                    for p in self.model_lm.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self.logger.log_metrics({'gradient_norm': total_norm}, train_step * simulated_batch_size)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        if not hasattr(self.scheduler,
                                       "total_steps") or train_step < self.scheduler.total_steps - 1:
                            self.scheduler.step()
                        self.logger.log_metrics({'lr': self.scheduler.get_lr()[0]},
                                                train_step * simulated_batch_size)

                    # Send train metrics to observers
                    self.logger.flush_batch_metrics(train_step * simulated_batch_size)

                    # Evaluate on validation set
                    if self.use_validation and validate_every and train_step % validate_every == 0:
                        t.measure()
                        self.model_lm.eval()
                        with torch.no_grad():
                            for validation_batch in islice(dataloader_validation, num_simulated_batches_valid):
                                validation_batch = batch_filter_distances(validation_batch, self.relative_distances)
                                validation_batch = batch_to_device(validation_batch, self.device)
                                output = self.model_lm.forward_batch(validation_batch).cpu()
                                validation_batch = batch_to_device(validation_batch, "cpu")

                                evaluation = self._evaluate_predictions(output.logits, validation_batch.labels,
                                                                        loss=output.loss, partition='valid')
                                self.logger.log_sub_batch_metrics(evaluation)

                                self.logger.log_text("predicted words/validation",
                                                     str(self._decode_predicted_words(output, validation_batch)))
                                self.logger.log_text("labels/validation",
                                                     str(self._decode_labels(validation_batch)))
                        self.model_lm.train()
                        self.logger.flush_batch_metrics(step=train_step * simulated_batch_size)
                        self.logger.log_time(t.measure() / simulated_batch_size_valid, "valid_seconds/sample",
                                             train_step * simulated_batch_size)

                if persistent_snapshot_every and (it + 1) % persistent_snapshot_every == 0:
                    snapshot_iteration = it + 1
                    self.logger.info(f"Storing model params into snapshot-{snapshot_iteration}")
                    self.model_manager.save_snapshot(run_id, self.model_lm.state_dict(), snapshot_iteration)
                    dataset = self.dataset_validation_creator(False)
                    score = self.evaluate(islice(dataset.to_dataloader(), int(max_validation_samples / batch_size)),
                                          train_step * simulated_batch_size, 'valid_full')
                    if f"micro_f1_score/valid_full" in self.logger.sub_batch_metrics:
                        score_name = 'micro-F1'
                    else:
                        score_name = 'F1'
                    self.logger.info(f"Full evaluation yielded {score} {score_name}")
                    if not early_stopping.evaluate(score, snapshot_iteration):
                        self.logger.info(f"Last {early_stopping_patience} evaluations did not improve performance. "
                                         f"Stopping run")

                        break

                progress_bar.update()
                if progress_bar.n >= progress_bar.total:
                    progress_bar = tqdm(total=int(self.data_manager.approximate_total_samples() / batch_size))
                    epoch += 1
                    progress_bar.set_description(f"Epoch {epoch}")

            t.measure()

        self._handle_shutdown()

    def _train_step(self, batch, num_simulated_batches):
        batch = batch_to_device(batch, self.device)
        output_gpu = self.model_lm.forward_batch(batch)
        # Gradient accumulation: every batch contributes only a part of the total gradient
        (output_gpu.loss / num_simulated_batches).backward()
        output_cpu = output_gpu.cpu()
        del output_gpu
        del batch

        return output_cpu

    def _evaluate_predictions(self, logits, labels, loss=None, partition='train'):
        evaluation = dict()
        for metric_name, metric_fn in self.metrics.items():
            evaluation[f"{metric_name}/{partition}"] = metric_fn(logits, labels)
        if loss:
            evaluation[f"loss/{partition}"] = loss.item()
        return evaluation

    def evaluate(self, dataset, step, partition='valid'):
        # Evaluate on validation set
        self.model_lm.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for validation_batch in dataset:
                validation_batch = batch_filter_distances(validation_batch, self.relative_distances)
                validation_batch = batch_to_device(validation_batch, self.device)
                output = self.model_lm.forward_batch(validation_batch).cpu()
                validation_batch = batch_to_device(validation_batch, "cpu")

                predictions.extend(output.logits.argmax(-1))
                labels.extend(validation_batch.labels)

                evaluation = self._evaluate_predictions(output.logits, validation_batch.labels,
                                                        loss=output.loss, partition=partition)
                self.logger.log_sub_batch_metrics(evaluation)

                self.logger.log_text("predicted words/validation",
                                     str(self._decode_predicted_words(output, validation_batch)))
                self.logger.log_text("labels/validation", str(self._decode_labels(validation_batch)))
        self.model_lm.train()
        if f"micro_f1_score/{partition}" in self.logger.sub_batch_metrics:
            score = mean(self.logger.sub_batch_metrics[f"micro_f1_score/{partition}"])
        else:
            score = mean(self.logger.sub_batch_metrics[f"f1_score/{partition}"])
        self.logger.flush_batch_metrics(step=step)

        return score

    def _decode_predicted_words(self, model_out, batch):
        method_name_vocab = self.method_name_vocab if self.use_separate_vocab else self.word_vocab
        if hasattr(self, 'use_pointer_network') and self.use_pointer_network:
            extended_vocab_reverse = {idx: word for word, idx in batch.extended_vocabulary[0].items()}
            predicted_sub_tokens = ((predicted_sub_token.argmax().item(), predicted_sub_token.max().item()) for
                                    predicted_sub_token in model_out.logits[0][0])
            return [
                (extended_vocab_reverse[st] if st in extended_vocab_reverse else method_name_vocab.reverse_lookup(st),
                 f"{value:0.2f}") for st, value in predicted_sub_tokens]
        else:
            return [(method_name_vocab.reverse_lookup(predicted_sub_token.argmax().item()),
                     f"{predicted_sub_token.max().item():0.2f}") for
                    predicted_sub_token in model_out.logits[0][0]]

    def _decode_labels(self, batch):
        method_name_vocab = self.method_name_vocab if self.use_separate_vocab else self.word_vocab
        if hasattr(self, 'use_pointer_network') and self.use_pointer_network:
            extended_vocab_reverse = {idx: word for word, idx in batch.extended_vocabulary[0].items()}
            label_tokens = (sub_token_label.item() for sub_token_label in batch.labels[0][0])
            return [extended_vocab_reverse[lt] if lt in extended_vocab_reverse else method_name_vocab.reverse_lookup(lt)
                    for lt in label_tokens]
        else:
            return [method_name_vocab.reverse_lookup(sub_token_label.item()) for sub_token_label in batch.labels[0][0]]

    def get_dataloader(self, split: str, batch_size: int):
        assert split == 'train' or split == 'validation'
        if split == 'train':
            ds = self.dataset_train
        elif split == 'validation':
            ds = self.dataset_validation
        dl = DataLoader(ds, batch_size=batch_size, num_workers=0,
                        collate_fn=ds.collate_fn)
        dl = DataLoaderWrapper(dl)
        return BufferedDataManager(dl)

    def _handle_shutdown(self, sig=None, frame=None):
        self.dataset_train.data_manager.shutdown()
        self.dataset_validation.data_manager.shutdown()
        sys.exit(0)


class EarlyStopping:

    def __init__(self, model_manager: ModelManager, run_id, patience):
        self.model_manager = model_manager
        self.run_id = run_id
        self.patience = patience
        self.evaluation_results = dict()
        self._counter = 0
        self._best = 0

    def evaluate(self, score, snapshot_iteration):
        self.evaluation_results[snapshot_iteration] = score
        sorted_results = sorted(self.evaluation_results.items(), key=lambda x: x[1], reverse=True)
        print(f"Current best performing snapshots: {sorted_results}")
        snapshots_to_keep = sorted_results[:self.patience]
        snapshots_to_keep = [x[0] for x in snapshots_to_keep]

        stored_snapshots = self.model_manager.get_available_snapshots(self.run_id)
        for stored_snapshot in stored_snapshots:
            if stored_snapshot not in snapshots_to_keep:
                self.model_manager.delete_snapshot(self.run_id, stored_snapshot)

        if score > self._best:
            self._best = score
            self._counter = 0
        else:
            self._counter += 1

        print(f"Counter: {self._counter}, Best: {self._best}")

        if self._counter > self.patience:
            return False
        else:
            return True
