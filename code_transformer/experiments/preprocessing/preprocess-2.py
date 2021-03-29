"""
File needs to be run with python -m scripts.run-preprocessing {config_file} {language} {partition} from the command line.
{config_file} is a .yaml file with configurations for stage2 preprocessing.
The script is intended to be run thrice for each {language}: 1. partition="train" 2./3. partition="valid"/"test"
During the first run on the train partition the vocabulary is generated. This vocabulary is then reused for
the validate and test partition
"""

import itertools
import signal
import sys
import traceback
from copy import deepcopy

import pandas as pd
from joblib import parallel_backend, Parallel, delayed

from sacred import Experiment

from code_transformer.modeling.constants import SPECIAL_SYMBOLS, SPECIAL_SYMBOLS_NODE_TOKEN_TYPES
from code_transformer.preprocessing.datamanager.base import CombinedDataManager
from code_transformer.preprocessing.datamanager.preprocessed import CTBufferedDataManager
from code_transformer.preprocessing.graph.binning import ExponentialBinning
from code_transformer.preprocessing.graph.distances import PersonalizedPageRank, ShortestPaths, AncestorShortestPaths, \
    SiblingShortestPaths, DistanceBinning
from code_transformer.preprocessing.graph.transform import DistancesTransformer
from code_transformer.preprocessing.nlp.vocab import WordCounter, CodeSummarizationVocabularyTransformer, \
    VocabularyTransformer
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Sample
from code_transformer.preprocessing.pipeline.stage2 import CTStage2MultiLanguageSample
from code_transformer.utils.log import get_logger
from code_transformer.utils.timing import Timing
from code_transformer.env import DATA_PATH_STAGE_1, DATA_PATH_STAGE_2

ex = Experiment(base_dir='../../..', interactive=False)


class Preprocess2Container:

    def __init__(self):
        self._init_execution()
        self._init_preprocessing()
        self._init_distances()
        self._init_binning()
        self._init_data()

        self.logger = get_logger(__file__)
        self._setup_data_managers()
        self._setup_vocabularies()
        self._setup_vocabulary_transformer()
        self._setup_distances_transformer()

        self._store_config()

    @ex.capture(prefix="execution")
    def _init_execution(self, num_processes, batch_size, dataset_slice_size):
        self.num_processes = num_processes
        self.batch_size = batch_size
        self.dataset_slice_size = dataset_slice_size

    @ex.capture(prefix="preprocessing")
    def _init_preprocessing(self,
                            remove_punctuation,
                            max_num_tokens,
                            vocab_size,
                            min_vocabulary_frequency,
                            separate_label_vocabulary,
                            vocab_size_labels,
                            min_vocabulary_frequency_labels):
        self.remove_punctuation = remove_punctuation
        self.max_num_tokens = max_num_tokens
        self.vocab_size = vocab_size
        self.min_vocabulary_frequency = min_vocabulary_frequency
        self.separate_label_vocabulary = separate_label_vocabulary
        self.vocab_size_labels = vocab_size_labels
        self.min_vocabulary_frequency_labels = min_vocabulary_frequency_labels

    @ex.capture(prefix="distances")
    def _init_distances(self,
                        ppr_alpha,
                        ppr_use_log,
                        ppr_threshold,
                        sp_threshold,
                        ancestor_sp_forward,
                        ancestor_sp_backward,
                        ancestor_sp_negative_reverse_dists,
                        ancestor_sp_threshold,
                        sibling_sp_forward,
                        sibling_sp_backward,
                        sibling_sp_negative_reverse_dists,
                        sibling_sp_threshold):
        self.ppr_alpha = ppr_alpha
        self.ppr_use_log = ppr_use_log
        self.ppr_threshold = ppr_threshold
        self.sp_threshold = sp_threshold
        self.ancestor_sp_forward = ancestor_sp_forward
        self.ancestor_sp_backward = ancestor_sp_backward
        self.ancestor_sp_negative_reverse_dists = ancestor_sp_negative_reverse_dists
        self.ancestor_sp_threshold = ancestor_sp_threshold
        self.sibling_sp_forward = sibling_sp_forward
        self.sibling_sp_backward = sibling_sp_backward
        self.sibling_sp_negative_reverse_dists = sibling_sp_negative_reverse_dists
        self.sibling_sp_threshold = sibling_sp_threshold

    @ex.capture(prefix='data')
    def _init_data(self, language, partition):
        self.language = language
        self.partition = partition
        self.use_multi_language = ',' in self.language
        if self.use_multi_language:
            self.languages = self.language.split(',')

        self.input_path = DATA_PATH_STAGE_1
        self.output_path = DATA_PATH_STAGE_2

    @ex.capture(prefix="binning")
    def _init_binning(self, num_bins,
                      n_fixed_bins,
                      exponential_binning,
                      exponential_binning_growth_factor,
                      bin_padding):
        self.num_bins = num_bins
        self.n_fixed_bins = n_fixed_bins
        self.exponential_binning = exponential_binning
        self.exponential_binning_growth_factor = exponential_binning_growth_factor
        self.bin_padding = bin_padding

    @ex.capture
    def _store_config(self, _config):
        config = deepcopy(_config)
        config['preprocessing']['special_symbols'] = SPECIAL_SYMBOLS
        config['preprocessing']['special_symbols_node_token_types'] = SPECIAL_SYMBOLS_NODE_TOKEN_TYPES
        self.output_data_manager.save_config(config)

    # =========================================================================
    # Setup Helper Methods
    # =========================================================================

    def _setup_data_managers(self):
        if self.use_multi_language:

            self.input_data_managers = [CTBufferedDataManager(self.input_path, l, self.partition) for l in
                                        self.languages]
            word_counters = zip(
                *[input_data_manager.load_word_counters() for input_data_manager in self.input_data_managers])

            word_counters = [
                self._combine_counters(word_counter, self.min_vocabulary_frequency)
                if i < 3 or not self.separate_label_vocabulary else
                self._combine_counters(word_counter, self.min_vocabulary_frequency_labels)
                for i, word_counter in enumerate(word_counters)]

            self.input_data_manager = CombinedDataManager(self.input_data_managers, self.languages)
        else:
            self.input_data_manager = CTBufferedDataManager(self.input_path, self.language, self.partition)
            word_counters = self.input_data_manager.load_word_counters()

        if self.separate_label_vocabulary:
            self.word_counter, self.token_type_counter, self.node_type_counter, self.word_counter_labels = word_counters
        else:
            self.word_counter, self.token_type_counter, self.node_type_counter = word_counters

        self.output_data_manager = CTBufferedDataManager(self.output_path, self.language, self.partition)

    def _setup_vocabularies(self):
        if self.partition == "train":
            # Only build vocabularies on train data
            self.word_vocab = self.word_counter.to_vocabulary(limit_most_common=self.vocab_size,
                                                              min_frequency=self.min_vocabulary_frequency,
                                                              special_symbols=SPECIAL_SYMBOLS)
            self.token_type_vocab = self.token_type_counter.to_vocabulary(
                special_symbols=SPECIAL_SYMBOLS_NODE_TOKEN_TYPES)
            self.node_type_vocab = self.node_type_counter.to_vocabulary(
                special_symbols=SPECIAL_SYMBOLS_NODE_TOKEN_TYPES)
            if self.separate_label_vocabulary:
                self.word_vocab_labels = self.word_counter_labels.to_vocabulary(
                    limit_most_common=self.vocab_size_labels,
                    min_frequency=self.min_vocabulary_frequency_labels,
                    special_symbols=SPECIAL_SYMBOLS)
        else:
            # On valid and test set, use already built vocabulary from train run
            if self.separate_label_vocabulary:
                self.word_vocab, self.token_type_vocab, self.node_type_vocab, self.word_vocab_labels = self.output_data_manager.load_vocabularies()
            else:
                self.word_vocab, self.token_type_vocab, self.node_type_vocab = self.output_data_manager.load_vocabularies()

    def _setup_vocabulary_transformer(self):
        if self.separate_label_vocabulary:
            self.vocabulary_transformer = CodeSummarizationVocabularyTransformer(self.word_vocab, self.token_type_vocab,
                                                                                 self.node_type_vocab,
                                                                                 self.word_vocab_labels)
        else:
            self.vocabulary_transformer = VocabularyTransformer(self.word_vocab, self.token_type_vocab,
                                                                self.node_type_vocab)

    def _setup_distances_transformer(self):
        distance_metrics = [
            PersonalizedPageRank(threshold=self.ppr_threshold, log=self.ppr_use_log, alpha=self.ppr_alpha),
            ShortestPaths(threshold=self.sp_threshold),
            AncestorShortestPaths(forward=self.ancestor_sp_forward, backward=self.ancestor_sp_backward,
                                  negative_reverse_dists=self.ancestor_sp_negative_reverse_dists,
                                  threshold=self.ancestor_sp_threshold),
            SiblingShortestPaths(forward=self.sibling_sp_forward, backward=self.sibling_sp_backward,
                                 negative_reverse_dists=self.sibling_sp_negative_reverse_dists,
                                 threshold=self.sibling_sp_threshold)]
        if self.exponential_binning:
            db = DistanceBinning(self.num_bins, self.n_fixed_bins,
                                 ExponentialBinning(self.exponential_binning_growth_factor))
        else:
            db = DistanceBinning(self.num_bins, self.n_fixed_bins)
        self.distances_transformer = DistancesTransformer(distance_metrics, db)

    def _combine_counters(self, counters, min_vocab_frequency):
        """
        If multiple languages are used, we need to combine the word counts for all of them.
        Additionally, if MIN_VOCABULARY_FREQUENCY is set, we only allow a token if if passed the
        threshold in ANY of the languages
        """

        combined_counter = WordCounter()

        # Dataframe with token x language -> count
        df = pd.DataFrame.from_dict(
            {language: counter.words for language, counter in zip(self.languages, counters)})
        df = df.fillna(0)
        if min_vocab_frequency is not None:
            idx_frequent_words = (df > min_vocab_frequency).any(axis=1)
            df = df[idx_frequent_words]
        df = df.sum(axis=1)

        combined_counter.words = df.to_dict()
        return combined_counter

    @staticmethod
    def _process_batch(x, vocabulary_transformer, distances_transformer, use_multi_language, max_num_tokens,
                       remove_punctuation):
        i, batch = x
        try:
            output = []
            for sample in batch:
                if use_multi_language:
                    sample_language, sample = sample
                assert len(sample) == 6, f"Unexpected sample format! {sample}"
                sample = CTStage1Sample.from_compressed(sample)
                if max_num_tokens is not None and len(sample.tokens) > max_num_tokens:
                    print(f"Snippet with {len(sample.tokens)} tokens exceeds limit of {max_num_tokens}! Skipping")
                    continue
                if remove_punctuation:
                    sample.remove_punctuation()
                sample = vocabulary_transformer(sample)
                sample = distances_transformer(sample)

                if use_multi_language:
                    sample = CTStage2MultiLanguageSample(sample.tokens, sample.graph_sample, sample.token_mapping,
                                                         sample.stripped_code_snippet, sample.func_name,
                                                         sample.docstring,
                                                         sample_language,
                                                         encoded_func_name=sample.encoded_func_name if hasattr(sample,
                                                                                                               'encoded_func_name') else None)

                output.append(sample)
            return output
        except Exception as e:
            # Cannot use logger in parallel worker, as loggers cannot be pickled
            print(str(e))
            traceback.print_exc()
            return []

    def _handle_shutdown(self, sig=None, frame=None):
        if self.use_multi_language:
            for idm in self.input_data_managers:
                idm.shutdown()
        else:
            self.input_data_manager.shutdown()
        self.output_data_manager.shutdown()
        sys.exit(0)

    def run(self):

        # -----------------------------------------------------------------------------
        # Multiprocessing Loop
        # -----------------------------------------------------------------------------

        self.logger.info("Start processing...")

        # Ensure graceful shutdown when preprocessing is interrupted
        signal.signal(signal.SIGINT, self._handle_shutdown)

        n_samples_after = 0
        with parallel_backend("loky") as parallel_config:
            execute_parallel = Parallel(self.num_processes, verbose=0)

            batched_samples_generator = enumerate(self.input_data_manager.read(self.batch_size))
            while True:
                dataset_slice = itertools.islice(batched_samples_generator,
                                                 int(self.dataset_slice_size / self.batch_size))
                with Timing() as t:
                    dataset = execute_parallel(delayed(self._process_batch)(batch,
                                                                            self.vocabulary_transformer,
                                                                            self.distances_transformer,
                                                                            self.use_multi_language,
                                                                            self.max_num_tokens,
                                                                            self.remove_punctuation)
                                               for batch in dataset_slice)

                if dataset:
                    dataset = [sample for batch in dataset for sample in batch]
                    n_samples_after += len(dataset)
                    self.logger.info(
                        f"processing {len(dataset)} samples took {t[0]:0.2f} seconds ({t[0] / len(dataset):0.3f} seconds "
                        f"per sample)")
                    self.output_data_manager.save(dataset)
                else:
                    break

        self.logger.info("Saving vocabulary")
        if self.separate_label_vocabulary:
            self.output_data_manager.save_vocabularies(self.word_vocab, self.token_type_vocab, self.node_type_vocab,
                                                       self.word_vocab_labels)
        else:
            self.output_data_manager.save_vocabularies(self.word_vocab, self.token_type_vocab, self.node_type_vocab)

        self.logger.info("PREPROCESS-2 DONE!")
        self.logger.info(
            f"Successfully processed {n_samples_after} samples")

        self._handle_shutdown()


@ex.automain
def main():
    Preprocess2Container().run()
