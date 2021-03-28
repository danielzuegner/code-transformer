import gzip
import json
import os
import pickle
from pathlib import Path


def save_zipped(obj, file):
    file = _file_ending(file, "p.gzip")
    create_directories(file)
    with gzip.open(file, 'wb') as f:
        pickle.dump(obj, f)


def load_zipped(file):
    file = _file_ending(file, "p.gzip")
    with gzip.open(file, 'rb') as f:
        return pickle.load(f)


def save_pickled(obj, file):
    file = _file_ending(file, "p")
    create_directories(file)
    with open(f"{file}", 'wb') as f:
        pickle.dump(obj, f)


def load_pickled(file):
    file = _file_ending(file, "p")
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_json(obj: dict, file):
    file = _file_ending(file, "json")
    create_directories(file)
    with open(file, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(file):
    file = _file_ending(file, "json")
    with open(file, 'r') as f:
        return json.load(f)


def _file_ending(file, ending):
    return f"{file}.{ending}" if f".{ending}" not in file else file


def create_directories(path):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
