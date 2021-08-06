import glob
import random
from copy import copy, deepcopy
from math import floor, ceil
from typing import List, Tuple

import numpy as np

from code_transformer.preprocessing.pipeline.stage2 import CTStage2MultiLanguageSample
from code_transformer.preprocessing.datamanager.base import DataManager, BufferedDataManager
from code_transformer.preprocessing.nlp.vocab import Vocabulary, WordCounter
from code_transformer.utils.io import save_zipped, load_zipped, save_json, load_json
from code_transformer.utils.log import get_logger

logger = get_logger(__file__)


class CTPreprocessedDataManager(DataManager):
    """
    The main data manager for preprocessed data. It takes care of folder structure, loading and saving dataset slices.
    """

    def __init__(self, data_location: str, language: str, partition="train", shuffle=False, infinite_loading=False,
                 mini_dataset=False, load_single_file=None, sort_by_length=False, chunk_size=None,
                 filter_language: str = None, dataset_imbalance: Tuple = None):
        """
        :param data_location: the main folder were samples will be loaded from and dataset slices saved to
        :param language: for which language samples should be loaded/saved
        :param partition: train, valid or test
        :param shuffle: whether load order should be randomized. Note: This is not a true randomization over all samples
        due to the limitations of not having random access on zipped files on disk. Thus, a best effort policy is
        pursued by shuffling the dataset files to read from and randomizing the samples within a dataset slice
        :param infinite_loading: if set to true, The data manager will provide infinite data by randomly sampling
        from the saved dataset slices every time instead of loading the list of files once at the beginning.
        Note: When using infinite loading it can happen that samples already occur a second time before the whole
        dataset has been seen once
        :param mini_dataset: if set to true, the data manager simulates a very small dataset that only consists of
        one dataset slice (roughly 5000 samples). The same dataset will always be used independent of `shuffle`. Can
        be used in combination with `infinite_loading`
        :param load_single_file: if set to a dataset file name, only this single dataset slice will be loaded
        :param sort_by_length: if set to true, loaded slices will be sorted by number of tokens. This can be useful
        if one wants to minimize the amount of zero-padding needed when batching several samples later on.
        :param chunk_size: If set to True and sort_by_length as well as shuffle are True, then the snippets (sorted by
        length) will be chunked into chunks of `chunk_size`, which will then be randomly shuffled.
        :param filter_language: Only applicable in a multilingual setting. If `language` contains multiple languages
        separated by comma then `filter_language` can be set to one of the languages specified in `language` to obtain
        only snippets of that language
        :param dataset_imbalance: Only applicable in a multilingual setting. If specified, should contain roughly the
        distribution of samples for each of the languages in `language` (in the same order). The data manager will then
        employ oversampling (i.e., duplication of samples from minority languages) to ensure that samples will occur
        evenly.
        """
        # Vocabularies and word counters are created on training data and used for all partitions
        if language not in {"poj_104", "codeforces"}:
            self.dataset_location = f"{data_location}/{language}/{partition}"
            self.vocabularies_path = f"{data_location}/{language}/vocabularies"
            self.word_counters_path = f"{data_location}/{language}/word-counters"
        else:
            self.dataset_location = f"{data_location}/{partition}"
            self.vocabularies_path = f"{data_location}/vocabularies"
            self.word_counters_path = f"{data_location}/word-counters"

        self.current_dataset_batch = 0
        self.language = language
        self.partition = partition
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.chunk_size = chunk_size
        assert not (self.shuffle and self.sort_by_length and self.chunk_size is None), \
            "incompatible combination of sorting and shuffling."
        self.infinite_loading = infinite_loading
        self.mini_dataset = mini_dataset
        self.load_single_file = load_single_file
        self.filter_language = filter_language

        assert filter_language is None or dataset_imbalance is None, f"Cannot specify both filter_language and" \
                                                                     f" dataset_imbalance"
        self.dataset_imbalance_multipliers = None
        if dataset_imbalance is not None:
            assert len(dataset_imbalance) == len(
                language.split(',')), f"Need to specify exactly one value per language " \
                                      f"for dataset_imbalance"
            majority_class_size = max(dataset_imbalance)
            self.dataset_imbalance_multipliers = [majority_class_size / lang_size for lang_size in dataset_imbalance]

    def save(self, dataset_slice: List, dataset_slice_idx=None, **kwargs):
        """
        Zips the dataset slice and writes it to disk. This is a rather time consuming operation.
        Dataset slices are automatically named in an ascending fashion.
        :param dataset_slice:
        :param dataset_slice_idx: directly specifies the name for the dataset slice to be saved. Necessary in a
        distributed setting where multiple nodes save into the same directory and the data manager is not shared
        """

        if dataset_slice_idx is None:
            dataset_slice_idx = self.current_dataset_batch
            self.current_dataset_batch += 1
        logger.info(f"Saving {len(dataset_slice)} samples into dataset-{dataset_slice_idx}.p.gzip")
        save_zipped(dataset_slice, f"{self.dataset_location}/dataset-{dataset_slice_idx}")

    def load_vocabularies(self):
        """
        Can only be used in stage 2.
        Returns a 3-tuple (word_vocab: Vocabulary, token_type_vocab: Vocabulary, node_type_vocab: Vocabulary)
        """

        return load_zipped(self.vocabularies_path)

    def save_vocabularies(self, word_vocab: Vocabulary, token_type_vocab: Vocabulary, node_type_vocab: Vocabulary,
                          word_vocab_labels: Vocabulary = None):
        """
        Can only be used in stage 2
        """

        if word_vocab_labels is None:
            save_zipped((word_vocab, token_type_vocab, node_type_vocab), self.vocabularies_path)
        else:
            save_zipped((word_vocab, token_type_vocab, node_type_vocab, word_vocab_labels), self.vocabularies_path)

    def load_word_counters(self):
        """
        Can only be used in stage 1
        Returns a 3-tuple (word_counter: WordCounter, token_type_counter: WordCounter, node_type_counter: WordCounter)
        or a 4-tuple (word_counter: WordCounter, token_type_counter: WordCounter, node_type_counter, word_counter_labels: WordCounter)
        """

        return load_zipped(self.word_counters_path)

    def save_word_counters(self, word_counter: WordCounter, token_type_counter: WordCounter,
                           node_type_counter: WordCounter, word_counter_labels: WordCounter = None):
        """
        Can only be used in stage 1
        """

        if word_counter_labels is None:
            # Combined vocabularies for input tokens and method name tokens
            save_zipped((word_counter, token_type_counter, node_type_counter), self.word_counters_path)
        else:
            # Separate vocabularies for input tokens and method name tokens
            save_zipped((word_counter, token_type_counter, node_type_counter, word_counter_labels),
                        self.word_counters_path)

    def save_config(self, config: dict):
        """
        Save the given preprocessing configuration
        """

        save_json(config, f"{self.dataset_location}/config")

    def load_config(self):
        """
        Load the preprocessing configuration that corresponds to this preprocessing run
        """

        return load_json(f"{self.dataset_location}/config")

    def approximate_total_samples(self, dataset_slice_size=None):
        if dataset_slice_size is None:
            config = self.load_config()
            if 'dataset_slice_size' in config['execution']:
                # Stage 2 dataset
                dataset_slice_size = config['execution']['dataset_slice_size']
            elif 'save_every' in config['execution']:
                # Stage 1 dataset
                dataset_slice_size = config['execution']['save_every']
            else:
                dataset_slice_size = 5000
        files = glob.glob(f"{self.dataset_location}/dataset-*.p.gzip")
        return (len(files) - 1) * dataset_slice_size + int(dataset_slice_size / 2)

    def _lazy_load_files(self):
        """
        Private generator for providing paths to dataset slices one by one. Implements shuffling of dataset slices.
        """

        # Prepare all dataset batches in data_location for lazy loading. This is necessary as loading everything at
        # once would not fit into the main memory.
        files = glob.glob(
            f"{self.dataset_location}/dataset-{'*' if self.load_single_file is None else self.load_single_file}.p.gzip")

        # Converts the file list into a generator
        if self.shuffle and not self.mini_dataset:
            random.shuffle(files)
        if not files:
            raise Exception(f"No dataset files found in {self.dataset_location}. Is the path correct?")

        if self.infinite_loading:
            if self.mini_dataset:
                while True:
                    yield files[0]
            else:
                while True:
                    yield random.choice(files)
        else:
            if self.mini_dataset:
                yield files[0]
            else:
                for file in files:
                    yield file

    def _load_zipped(self, file):
        """
        Loads and unzips the given file. Shuffles the samples within the dataset slice if wished.
        :return: samples of the dataset slice
        """

        logger.info(f"Loading {file}...")
        data = load_zipped(file)

        if self.filter_language:
            assert isinstance(data[0],
                              CTStage2MultiLanguageSample), f"filter_language can only be used on multilingual corpora"
            data = [sample for sample in data if sample.language == self.filter_language]
        elif self.dataset_imbalance_multipliers:
            assert isinstance(data[0],
                              CTStage2MultiLanguageSample), f"dataset_imbalance can only be used on multilingual " \
                                                             f"corpora"
            duplicated_data = []
            language_mapping = {lang: lang_idx for lang_idx, lang in enumerate(self.language.split(','))}
            for sample in data:
                lang_idx = language_mapping[sample.language]
                imbalance_multiplier = self.dataset_imbalance_multipliers[lang_idx]
                # If imbalance multiplier is not an integer, we need to sample an integer in such a way, to ensure that
                # overall, the sample values correspond to the continuous imbalance multiplier
                if floor(imbalance_multiplier) != ceil(imbalance_multiplier):
                    p = imbalance_multiplier - floor(imbalance_multiplier)
                    imbalance_multiplier = np.random.choice([floor(imbalance_multiplier), ceil(imbalance_multiplier)],
                                                            p=(1 - p, p))
                imbalance_multiplier = int(imbalance_multiplier)
                duplicated_data.extend([deepcopy(sample) for _ in range(imbalance_multiplier)])
            data = duplicated_data

        if self.sort_by_length:
            data = sorted(data, key=lambda x: len(x.tokens))
            if self.shuffle and self.chunk_size is not None:
                chunked = list(chunker(data, self.chunk_size))
                random.shuffle(chunked)
                data = [y for x in chunked for y in x]
        elif self.shuffle:
            random.shuffle(data)

        logger.info(f"Loaded {len(data)} samples")
        return data

    def __iter__(self):
        """
        Returns a generator for all samples in the dataset. A generator is chosen here to allow lazy unzipping of
        dataset batches once the samples are actually needed.
        """

        # Initialize internal files generator
        lazy_load_files = self._lazy_load_files()
        # Samples are directly drawn from files and files are loaded last minute ad hoc
        sample_generator = (sample for file in lazy_load_files for sample in self._load_zipped(file))
        return sample_generator


class CTBufferedDataManager(CTPreprocessedDataManager, BufferedDataManager):
    """
    Convenience class that wraps a BufferedDataManager around a CTPreprocessedDataManager. That way,
    a CTBufferedDataManager simply inherits all methods and properties from both classes.
    """

    def __init__(self, data_location: str, language: str, partition="train", shuffle=False, sort_by_length=False,
                 size_load_buffer=5000,
                 size_save_buffer=1, infinite_loading=False, mini_dataset=False, chunk_size=None,
                 filter_language: str = None, dataset_imbalance: Tuple = None):
        CTPreprocessedDataManager.__init__(self, data_location, language, partition, shuffle, infinite_loading,
                                           mini_dataset, sort_by_length=sort_by_length, chunk_size=chunk_size,
                                           filter_language=filter_language, dataset_imbalance=dataset_imbalance)
        # Casting self to CTPreprocessedDataManager such that BufferedDataManager uses the CTPreprocessedDataManager's
        # __iter__ and __next__ functions and not the ones defined here (which would lead to infinite recursion
        data_manager = copy(self)
        data_manager.__class__ = CTPreprocessedDataManager
        BufferedDataManager.__init__(self, data_manager, size_load_buffer, size_save_buffer)

    def __iter__(self):
        """
        Calls the BufferedDataManager's .__iter__() which will then utilize the CTPreprocessedDataManager's .__iter__()
        """

        return BufferedDataManager.__iter__(self)

    def save(self, dataset_slice: List, **kwargs):
        """
        Calls the BufferedDataManager's .save() which will then utilize the CTPreprocessedDataManager's .save()
        """

        return BufferedDataManager.save(self, dataset_slice)

    def __del__(self):
        BufferedDataManager.__del__(self)


def chunker(seq, size):
    """
    Chunk a list into chunks of size `size`.
    From
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    Parameters
    ----------
    seq: input list
    size: size of chunks
    Returns
    -------
    The list of lists of size `size`
    """

    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
