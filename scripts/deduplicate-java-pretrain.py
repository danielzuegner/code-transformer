"""
To generate the huge java-pretrain dataset, we first manually merge java-small, java-medium and java-large into the
same folder. It can then happen that in the new larger training set there are Java methods from the same projects or
even the same methods as in the test partition for java-small for example. This would skew the final results and
violate the underlying design principle behind the code2seq datasets to split the partitions by project.
Hence, in this script we search for similar pairs of java classes between the train partition of java-pretrain and
the valid/test partition of java-small/java-medium.

Similarity is defined as follows:
 1) Only pairs of Java class with the same file name are considered, otherwise comparing every possible pair kills us.
    Usually, if two Java classes are the same or have only been altered slightly, their file names will be the same.
 2) Similarity of a candidate pair is computed using difflib's SequenceMatcher
 3) If similarity exceeds a certain threshold (0.7 in our experiments) the corresponding file is marked to be  deleted
    from the train partition of java-pretrain

This script does not delete files, it only computes the list of files to be deleted and stores it in
`{CODE2SEQ_RAW_DATA_PATH}/java-pretrain/files_to_delete_{dataset}_{partition}.p`

This pickled file can then be loaded in the deduplication notebook where further project-level deletion is done to
minimize the risk of having Java methods from the respective code2seq valid/test partitions in the new java-pretrain
train partition.
"""

import argparse
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from joblib import parallel_backend, Parallel, delayed
from tqdm import tqdm

from code_transformer.preprocessing.datamanager.base import DataManager
from code_transformer.utils.io import save_pickled
from code_transformer.env import CODE2SEQ_RAW_DATA_PATH

FILE_SIMILARTY = 0.7  # Similarity threshold that defines when to delete a file in java-pretrain
NUM_PROCESSES = 12
BATCH_SIZE = 5

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["java-small", "java-medium"])
parser.add_argument("partition", choices=["validation", "test"])

args = parser.parse_args()

data_path_code2seq = CODE2SEQ_RAW_DATA_PATH

dataset = args.dataset
partition = args.partition
projects_folder = Path(f"{data_path_code2seq}/{dataset}/{partition}")

reference_dataset = 'java-pretrain'
reference_partition = 'training'
reference_projects_folder = Path(f"{data_path_code2seq}/{reference_dataset}/{reference_partition}")
reference_projects = {p for p in reference_projects_folder.iterdir()}


def get_files_recursive(folder):
    if not folder.is_dir():
        return [folder]
    results = []
    for file in folder.iterdir():
        if file.is_dir():
            results.extend(get_files_recursive(file))
        else:
            results.append(file)
    return results


def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)


def read_file(f):
    try:
        return f.read_text()
    except UnicodeDecodeError:
        return f.read_text(encoding='cp1252')


def similar(a, b):
    return SequenceMatcher(None, a, remove_comments(read_file(b))).ratio()


def find_files_to_delete(batch):
    files_to_delete = []
    for search_file, candidate_files in batch:
        try:
            search_file_content = remove_comments(search_file.read_text())
        except UnicodeDecodeError:
            continue
        for candidate_file in candidate_files:
            try:
                if similar(search_file_content, candidate_file) > FILE_SIMILARTY:
                    files_to_delete.append(candidate_file)
            except UnicodeDecodeError:
                pass
    return files_to_delete


if __name__ == '__main__':
    print("Indexing files...")
    file_lookup = defaultdict(list)

    for i, p in enumerate(reference_projects_folder.iterdir()):
        for f in get_files_recursive(p):
            file_lookup[f.stem].append(f)

    results = dict()
    files_to_delete = []

    with parallel_backend("loky") as parallel_config:
        execute_parallel = Parallel(NUM_PROCESSES, verbose=0)

        print(len(list(projects_folder.iterdir())))

        for i, project in enumerate(tqdm(list(projects_folder.iterdir()))):
            search_files = get_files_recursive(project)
            num_files = len(search_files)
            print(project.stem, num_files)
            batch_generator = DataManager.to_batches(((search_file, file_lookup[search_file.stem]) for search_file in
                                                      search_files),
                                                     BATCH_SIZE)
            result = execute_parallel(
                delayed(find_files_to_delete)(batch) for batch in batch_generator)
            for res in result:
                files_to_delete.extend(res)

        save_pickled(set(files_to_delete), f"{data_path_code2seq}/java-pretrain/files_to_delete_{dataset}_{partition}")
