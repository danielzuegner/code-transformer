"""
File needs to be run with python -m scripts.run-preprocessing {config_file} {language} {partition} from the command line.
{config_file} is a .yaml file with configurations for stage1 preprocessing.
The script is intended to be run thrice for each {language}: 1. partition="train" 2./3. partition="valid"/"test"
"""

import itertools
import os
import random
import signal
import sys
import traceback

from joblib import parallel_backend, Parallel, delayed
from sacred import Experiment

from code_transformer.preprocessing.datamanager.c2s.raw import C2SRawDataLoader
from code_transformer.preprocessing.datamanager.csn.raw import CSNRawDataLoader
from code_transformer.preprocessing.datamanager.cpp.raw import CPPRawDataLoader
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.nlp.vocab import WordCounter, CodeSummarizationVocabularyBuilder, VocabularyBuilder
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Preprocessor, PreprocessingException
from code_transformer.utils.log import get_logger
from code_transformer.utils.timing import Timing
from code_transformer.env import CODE2SEQ_EXTRACTED_METHODS_DATA_PATH, CPP_RAW_DATA_PATH, \
    CSN_RAW_DATA_PATH, DATA_PATH_STAGE_1

ex = Experiment(base_dir='../../..', interactive=False)


class Preprocess1Container:

    def __init__(self):
        self._init_execution()
        self._init_preprocessing()
        self._init_data()

        self.logger = get_logger(__file__)
        random.seed(self.random_seed)

        self._setup_data_manager()
        self._setup_vocab_builder()
        self._setup_preprocessor()
        self._setup_data_loader()
        self._store_config()

    # =========================================================================
    # Parameter initializations
    # =========================================================================

    @ex.capture(prefix='execution')
    def _init_execution(self, num_processes, batch_size, save_every, random_seed):
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.save_every = save_every
        self.random_seed = random_seed

    @ex.capture(prefix='preprocessing')
    def _init_preprocessing(self, hard_num_tokens_limit, allow_empty_methods, use_tokens_limiter,
                            separate_label_vocabulary):
        self.hard_num_tokens_limit = hard_num_tokens_limit
        self.allow_empty_methods = allow_empty_methods
        self.use_tokens_limiter = use_tokens_limiter
        self.separate_label_vocabulary = separate_label_vocabulary

    @ex.capture(prefix='data')
    def _init_data(self, language, partition):
        self.partition = partition

        if language in {'java-small', 'java-medium', 'java-large'}:
            self.dataset_name = language
            self.language = "java"
            self.dataset_type = "code2seq"
        elif language == "cpp":
            self.dataset_name = language
            self.language = "cpp"
            self.dataset_type = "cpp"
        else:
            self.dataset_name = language
            self.dataset_type = "code-search-net"
            self.language = language

        if self.dataset_type == 'code2seq':
            self.input_data_path = CODE2SEQ_EXTRACTED_METHODS_DATA_PATH
        elif self.dataset_type == 'cpp':
            self.input_data_path = CPP_RAW_DATA_PATH
        else:
            self.input_data_path = CSN_RAW_DATA_PATH

    @ex.capture
    def _store_config(self, _config):
        self.data_manager.save_config(_config)

    # =========================================================================
    # Setup Helper methods
    # =========================================================================

    def _setup_data_manager(self):
        self.data_manager = CTBufferedDataManager(DATA_PATH_STAGE_1, self.dataset_name, self.partition)

    def _setup_vocab_builder(self):
        if self.partition == 'train':
            word_counter = WordCounter()
            token_type_counter = WordCounter()
            node_type_counter = WordCounter()
            if self.separate_label_vocabulary:
                word_counter_labels = WordCounter()
                self.word_counters = (word_counter, token_type_counter, node_type_counter, word_counter_labels)
                self.vocab_builder = CodeSummarizationVocabularyBuilder(*self.word_counters)

            else:
                self.word_counters = (word_counter, token_type_counter, node_type_counter)
                self.vocab_builder = VocabularyBuilder(*self.word_counters)

    def _setup_preprocessor(self):
        self.preprocessor = CTStage1Preprocessor(self.language,
                                                 allow_empty_methods=self.allow_empty_methods,
                                                 use_tokens_limiter=self.use_tokens_limiter,
                                                 max_num_tokens=self.hard_num_tokens_limit)

    def _setup_data_loader(self):
        self.logger.info(f"loading dataset {self.dataset_name} for {self.language}...")
        if self.dataset_type == 'code2seq':
            self.dataloader = C2SRawDataLoader(self.input_data_path)
            self.dataloader.load_dataset(self.dataset_name, partition=self.partition)
        elif self.dataset_type == "cpp":
            self.dataloader = CPPRawDataLoader(self.input_data_path)
            self.dataloader.load_all_for(partition=self.partition)
        else:
            self.dataloader = CSNRawDataLoader(self.input_data_path)
            self.dataloader.load_all_for(self.language, partition=self.partition)

        self.n_raw_samples = len(self.dataloader)
        self.logger.info(f"Loaded {self.n_raw_samples} snippets")

    # =========================================================================
    # Processing Helper Methods
    # =========================================================================

    @staticmethod
    def _process_batch(preprocessor, x):
        i, batch = x
        try:
            return preprocessor.process(batch, i)
        except PreprocessingException as e:
            # Cannot use logger in parallel worker, as loggers cannot be pickled
            print(str(e))
            # This is an expected exception, thus we just return an empty list, such that preprocessing can go on
            return []
        except Exception as e:
            print(f"Error processing batch {i}:")
            func_names, docstrings, code_snippets = zip(*batch)
            print(str(e))
            for snippet in code_snippets:
                print(snippet)
            traceback.print_exc()
            return []

    def _save_dataset(self, dataset):
        # Building vocabulary before saving. Ensures that it is run in main process => no race conditions when updating
        # vocabulary
        if self.partition == 'train':
            for sample in dataset:
                self.vocab_builder(sample)
        dataset = [sample.compress() for sample in dataset]
        if dataset:
            self.logger.info(f"saving dataset batch with {len(dataset)} samples ...")
            self.data_manager.save(dataset)

    def _handle_shutdown(self, sig=None, frame=None):
        self.data_manager.shutdown()

    # =========================================================================
    # Main method
    # =========================================================================

    def run(self):
        os.umask(0o007)

        # Ensure graceful shutdown when preprocessing is interrupted
        signal.signal(signal.SIGINT, self._handle_shutdown)

        n_processed_samples = 0
        with parallel_backend("loky") as parallel_config:
            execute_parallel = Parallel(self.num_processes, verbose=0)
            batched_samples_generator = enumerate(self.dataloader.read(self.batch_size, shuffle=True))
            while True:
                self.logger.info("start processing batch ...")
                dataset_slice = itertools.islice(batched_samples_generator, int(self.save_every / self.batch_size))
                with Timing() as t:
                    dataset = execute_parallel(delayed(self._process_batch)(self.preprocessor, batch) for batch in dataset_slice)

                if dataset:
                    dataset = [sample for batch in dataset for sample in batch]  # List[batches] -> List[samples]
                    self.logger.info(
                        f"processing {len(dataset)} samples took {t[0]:0.2f} seconds ({t[0] / len(dataset):0.3f} seconds per "
                        f"sample)")
                    self._save_dataset(dataset)
                    n_processed_samples += len(dataset)
                else:
                    break

        if self.partition == 'train':
            self.data_manager.save_word_counters(*self.word_counters)

        self.logger.info("PREPROCESS-1 DONE!")
        self.logger.info(
            f"Successfully processed {n_processed_samples}/{self.n_raw_samples} samples ({n_processed_samples / self.n_raw_samples:0.2%})")

        self._handle_shutdown()


@ex.automain
def main():
    Preprocess1Container().run()
