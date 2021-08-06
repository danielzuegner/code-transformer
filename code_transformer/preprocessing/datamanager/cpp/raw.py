"""
Dataloader for the raw method snippets from the CPP dataset stored as raw cpp files.
"""

import os
import random
from os.path import dirname, basename

from code_transformer.preprocessing.datamanager.base import DataManager, RawDataLoader
from code_transformer.preprocessing.datamanager.csn.raw import CSNRawSample


class CPPRawDataLoader(RawDataLoader):
    """
    Loads the code snippets from the CPP dataset.
    Returns sample as a 3-tuple containing
    """

    def __init__(self, data_location):
        self.data_location = data_location
        self.lines = []

    def load(self, file_path_):
        with open(file_path_, 'r', encoding="ascii") as f:
            code = f.read()
        label = basename(dirname(file_path_))
        self.lines.append((label, code))

    def load_all_for(self, partition=None):
        if partition is None:
            partitions = ["train", "valid", "test"]
        else:
            partitions = [partition]
        for part in partitions:
            path_ = f"{self.data_location}/{part}"
            for dir_ in os.listdir(path_):
                dir_path_ = os.path.join(path_, dir_)
                if os.path.isdir(dir_path_):
                    for file_path_ in os.listdir(dir_path_):
                        self.load(os.path.join(dir_path_, file_path_))

    def get_available_languages(self):
        return [name for name in os.listdir(self.data_location) if os.path.isdir(f"{self.data_location}/{name}")]

    def __len__(self):
        return len(self.lines)

    def read(self, batch_size=1, shuffle=False):
        if shuffle:
            lines = random.sample(self.lines, len(self.lines))
        else:
            lines = self.lines

        # Each line is a tuple (label, code)
        reader = map(lambda line: CSNRawSample(line[0], None, line[1]), lines)

        if batch_size > 1:
            return DataManager.to_batches(reader, batch_size)
        else:
            return reader
