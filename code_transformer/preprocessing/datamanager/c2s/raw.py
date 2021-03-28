"""
Loader class to facilitate loading code2seq .json dataset files that were generated from raw java classes by using
the JavaMethodExtractor.jar
"""

import glob
import json
import os
import random

from code_transformer.preprocessing.datamanager.base import DataManager, RawDataLoader
from code_transformer.preprocessing.datamanager.csn.raw import CSNRawSample


class C2SRawDataLoader(RawDataLoader):

    def __init__(self, data_location):
        self.data_location = data_location

    def get_available_datasets(self):
        return [name for name in os.listdir(self.data_location) if os.path.isdir(f"{self.data_location}/{name}")]

    def load_dataset(self, dataset, partition="train"):
        if os.path.isdir(f"{self.data_location}/{dataset}/{partition}"):
            # raw methods are separated in multiple dataset slices
            files = glob.glob(f"{self.data_location}/{dataset}/{partition}/dataset-*.json")
            self.samples = []
            for file in files:
                with open(file, 'r') as f:
                    self.samples.extend(json.load(f))
        else:
            with open(f"{self.data_location}/{dataset}/{partition}.json", 'r') as f:
                self.samples = json.load(f)

    def read(self, batch_size=1, shuffle=False):
        if shuffle:
            lines = random.sample(self.samples, len(self.samples))
        else:
            lines = self.samples

        reader = map(lambda sample: CSNRawSample(sample['name'], sample['doc'] if 'doc' in sample else None,
                                                 sample['code']), lines)

        if batch_size > 1:
            return DataManager.to_batches(reader, batch_size)
        else:
            return reader

    def __len__(self):
        return len(self.samples)
