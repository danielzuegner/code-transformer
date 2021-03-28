"""
Dataloader for the raw method snippets from the CodeSearchNet dataset stored as .jsonl.gz files.
"""

import gzip
import os
from collections import namedtuple
import random

import jsonlines

from code_transformer.preprocessing.datamanager.base import DataManager, RawDataLoader

CSNRawSample = namedtuple("CSNRawSample", ["func_name", "docstring", "code_snippet"])


class CSNRawDataLoader(RawDataLoader):
    """
    Loads and unzips the code snippets from the Code Search Net dataset.
    Returns sample as a 3-tuple containing
    """

    def __init__(self, data_location):
        self.data_location = data_location
        self.lines = []

    def load(self, language, partition, seq):
        with gzip.GzipFile(
                f"{self.data_location}/{language}/final/jsonl/{partition}/{language}_{partition}_{seq}.jsonl.gz",
                'r') as fin:
            json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        self.lines.extend(json_str.split("\n"))

    def load_all_for(self, language, partition=None):
        if partition is None:
            partitions = ["train", "valid", "test"]
        else:
            partitions = [partition]
        for part in partitions:
            for seq in range(self.get_num_files(language, part)):
                self.load(language, part, seq)

    def get_available_languages(self):
        return [name for name in os.listdir(self.data_location) if os.path.isdir(f"{self.data_location}/{name}")]

    def get_num_files(self, language, partition):
        return len(os.listdir(f"{self.data_location}/{language}/final/jsonl/{partition}"))

    def __len__(self):
        return len(self.lines)

    def read(self, batch_size=1, shuffle=False):
        if shuffle:
            lines = random.sample(self.lines, len(self.lines))
        else:
            lines = self.lines
        reader = jsonlines.Reader(lines)
        reader = map(lambda line: CSNRawSample(line['func_name'], line['docstring'], line['code']), reader)

        if batch_size > 1:
            return DataManager.to_batches(reader, batch_size)
        else:
            return reader
