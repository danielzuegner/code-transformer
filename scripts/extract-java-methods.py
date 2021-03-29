"""
Python wrapper for the JavaMethodExtractor to extract code snippets containing Java methods from the code2seq datasets.
After this script is run, the extracted methods can be further preprocessed with our stage1 and stage2 pipeline to be
fed into a CodeTransformer model eventually.
"""

import argparse
import subprocess

from code_transformer.env import CODE2SEQ_RAW_DATA_PATH, CODE2SEQ_EXTRACTED_METHODS_DATA_PATH, JAVA_METHOD_EXTRACTOR_EXECUTABLE, \
      JAVA_EXECUTABLE

parser = argparse.ArgumentParser()
parser.add_argument("language")
args = parser.parse_args()

cmd_train = f"{JAVA_EXECUTABLE} -jar {JAVA_METHOD_EXTRACTOR_EXECUTABLE} " \
            f"--dir {CODE2SEQ_RAW_DATA_PATH}/{args.language}/training " \
            f"--output_dir {CODE2SEQ_EXTRACTED_METHODS_DATA_PATH}/{args.language}/train"

subprocess.check_call(cmd_train, shell=True)

cmd_valid = f"{JAVA_EXECUTABLE} -jar {JAVA_METHOD_EXTRACTOR_EXECUTABLE} " \
            f"--dir {CODE2SEQ_RAW_DATA_PATH}/{args.language}/validation " \
            f"--output_dir {CODE2SEQ_EXTRACTED_METHODS_DATA_PATH}/{args.language}/valid"

subprocess.check_call(cmd_valid, shell=True)

cmd_test = f"{JAVA_EXECUTABLE} -jar {JAVA_METHOD_EXTRACTOR_EXECUTABLE} " \
           f"--dir {CODE2SEQ_RAW_DATA_PATH}/{args.language}/test " \
           f"--output_dir {CODE2SEQ_EXTRACTED_METHODS_DATA_PATH}/{args.language}/test"

subprocess.check_call(cmd_test, shell=True)
