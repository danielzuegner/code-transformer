"""
Executes stage 1 and stage 2 data preprocessing of code snippets.
Usage: python -m scripts.run-preprocessing {config_file} {language} {train|valid|test}
the {config_file} is a .yaml file that contains preprocessing-specific configuration.
See code_transformer/experiments/preprocessing/preprocess-1.yaml for an example.
"""

import argparse
import subprocess

from code_transformer.utils.sacred import read_config, parse_command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("language")
    parser.add_argument("partition", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = read_config(args.config_file)
    config['data'] = dict(language=args.language,
                          partition=args.partition)
    exe, cmd = parse_command(config)

    subprocess.check_call(cmd, shell=True)
