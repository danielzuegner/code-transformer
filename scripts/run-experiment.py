"""
Starts a train run.
Usage: python -m scripts.run-experiments {config_file}
the {config_file} is a .yaml file that contains experiment-specific configuration.
See code_transformer/experiments/code_transformer/code_summarization.yaml for an example.
"""

import argparse
import subprocess

from code_transformer.utils.sacred import read_config, parse_command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    config = read_config(args.config_file)
    exe, cmd = parse_command(config)

    subprocess.check_call(cmd, shell=True)
