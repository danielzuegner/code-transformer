import re
import sys
import os
import logging
from glob import glob
from pathlib import Path


def get_logger(file):
    return Logger(os.path.basename(file))


class Logger(logging.Logger):

    def __init__(self, name):
        super(Logger, self).__init__(name, logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        handler.flush = sys.stdout.flush
        super(Logger, self).addHandler(handler)


def generate_run_name(run_dir, run_prefix, match_arbitrary_suffixes=False):
    """
    Generates a new run name by searching for existing runs and adding 1 to the one with the highest ID.
    Assumes that runs will be stored in folder run_dir and have format "{run_prefix}-{run_id}".
    If `arbitrary_suffixes` = True the assumed format is less strict, i.e., folders/files in `run_dir`will be counted if
    they have the format "{run_prefix}-{run_id}*" instead.

    Parameters
    ----------
    run_dir:
        In which folders the runs are stored
    run_prefix:
        Prefix assumed for the run names. Searches for names with format "{run_prefix}-{run_id}"
    match_arbitrary_suffixes
        If set, searches for "{run_prefix}-{run_id}*" instead

    Returns
    -------
        A run name with format "{run_prefix}_{run_id}" where run_id is one larger than what is already found in `run_dir`
    """

    if match_arbitrary_suffixes:
        regex_string = rf"{run_prefix}-(\d+)"
    else:
        regex_string = rf"{run_prefix}-(\d+)$"
    regex = re.compile(regex_string)
    run_names = glob(f"{run_dir}/{run_prefix}-*")
    run_names = [Path(run_name).stem for run_name in run_names]
    run_ids = [int(regex.search(run_name).group(1)) for run_name in run_names if regex.match(run_name)]

    run_id = max(run_ids) + 1 if len(run_ids) > 0 else 1
    return f"{run_prefix}-{run_id}"


def list_file_numbering(directory, prefix, suffix=None):
    """
    Finds all files in the specified directory that match the given {prefix}{number}{suffix} pattern.
    All found {number}s are returned as a list in ascending order.

    Parameters
    ----------
        directory: where to search for file numberings
        prefix: prefix of files to be considered
        suffix: (optional) suffix of files to be considered

    Returns
    -------
        a list of numbers (without leading zeros) that appear in the matched file names in between `prefix` and `suffix`.
    """
    if suffix is None:
        suffix = ""
    regex = re.compile(f"{prefix}-(-?\d+)$")
    file_names = glob(f"{directory}/{prefix}-*{suffix}")
    file_names = [Path(file_name).stem for file_name in file_names]
    numbering = sorted([int(regex.search(file_name).group(1)) for file_name in file_names if regex.match(file_name)])
    return numbering
