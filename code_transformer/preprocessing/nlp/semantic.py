"""
Uses semantic to obtain an AST for a given code snippet in a language supported by semantic.
This code is based on a semantic version that still supports the `--json-graph` option! Newer versions of semantic
have dropped that command line option and it is unclear if it will return! To ensure this code will work with semantic,
build a semantic from a revision before Mar 27, 2020! E.g.,
https://github.com/github/semantic/tree/34ea0d1dd6ac1a142e2215f097f17abeed66de34
"""

import glob
import json
import os
import shutil
import subprocess
import threading
from collections import OrderedDict
from json import JSONDecodeError

from code_transformer.env import SEMANTIC_EXECUTABLE
from code_transformer.utils.log import get_logger

logger = get_logger(__file__)

TEMP_PIPE = "/tmp/semantic-temp-pipe"
SEMANTIC_CMD = [SEMANTIC_EXECUTABLE]
if shutil.which(" ".join(SEMANTIC_CMD)) is None:
    assert shutil.which("semantic") is not None, f"Could not locate semantic executable in {SEMANTIC_CMD}! Is the path correct?"
    logger.warn(f"Could not locate semantic executable in {SEMANTIC_CMD}! Falling back to semantic executable found "
                f"on PATH")
    SEMANTIC_CMD = ["semantic"]

language_file_extensions = {
    "python": "py",
    "javascript": "js",
    "ruby": "rb",
    "typescript": "ts",
    "go": "go",
    "json": "json",
    "jsx": "jsx",
    "php": "php"
}


def run_semantic(command, arg, output_type, *files, quiet=True):
    assert shutil.which(" ".join(SEMANTIC_CMD)) is not None, f"Could not locate semantic executable in {SEMANTIC_CMD}! Is the path correct?"

    call = []
    call.extend(SEMANTIC_CMD)
    call.append(arg)
    call.extend([command, output_type])
    call.extend(files)
    call[0] = call[0].replace(' ', '\\ ')  # Ensure all whitespaces in the path are escaped
    cabal_call = subprocess.Popen(' '.join(call), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, shell=True)
    output, errors = cabal_call.communicate()

    cabal_call.wait()

    # if output of a command is valid json, already parse it
    if output_type == '--json' or output_type == '--json-graph' or output_type == '--symbols':
        try:
            output = json.loads(output, object_pairs_hook=OrderedDict)
        except JSONDecodeError as e:
            raise ValueError(f"semantic command `{' '.join(call)}` "
                             f"returned JSON `{output}`") from e

    if not errors == "":
        # We can filter the erroneous files
        successful_parses = []
        successful_runs = []
        if output_type == '--json-graph':
            for i, file in enumerate(output['files']):
                if 'errors' in file:
                    if not quiet:
                        for error in file['errors']:
                            logger.error(f"{file['path']}: {error['error']}")
                elif 'Error' not in [vertex['term'] for vertex in file['vertices']]:
                    successful_parses.append(file)
                    successful_runs.append(i)
            # we need to return the indices for successful parses for the caller
            return {'files': successful_parses}, successful_runs
        else:
            raise Exception(errors)

    # Returning None as successful_runs means that all runs were successful
    return output, None


def run_semantic_parse(arg, output_type, *files, quiet=True):
    return run_semantic("parse", arg, output_type, *files, quiet=quiet)


def semantic_parse(language, arg, output_type, process_identifier, *code_snippets, quiet=True):
    """
    Semantic only accepts files as input. To avoid unnecessary disk I/O, we create a named pipe (TEMP_PIPE) that is used for writing.
    Semantic will then receive the named pipe and read from it as it was a file. Due to the nature of pipes where a write hangs until someone reads from the pipe,
    we need an async write.
    """

    def pipe_writer_worker(code, pipe_name):
        with open(pipe_name, 'w') as temp_pipe:
            temp_pipe.write(code)

    if language not in language_file_extensions:
        raise Exception(f"language `{language}` not supported by semantic")

    if not isinstance(code_snippets, list) and not isinstance(code_snippets, tuple) and not isinstance(code_snippets,
                                                                                                       set):
        code_snippets = [code_snippets]

    file_extension = language_file_extensions[language]

    # Create temporary pipes
    pipes_wildcard = f"{TEMP_PIPE}-{process_identifier}-*.{file_extension}"
    cleanup_temp_pipes(process_identifier, file_extension)
    for i, code in enumerate(code_snippets):
        pipe_name = f"{TEMP_PIPE}-{process_identifier}-{i:05d}.{file_extension}"
        if not os.path.exists(pipe_name):
            os.mkfifo(pipe_name)

        # Write to pipe asynchroneously 
        threading.Thread(target=pipe_writer_worker, args=(code, pipe_name)).start()

    result = run_semantic_parse(arg, output_type, pipes_wildcard, quiet=quiet)
    cleanup_temp_pipes(process_identifier, file_extension)

    return result


def cleanup_temp_pipes(process_identifier, file_extension):
    pipes_wildcard = f"{TEMP_PIPE}-{process_identifier}-*.{file_extension}"
    if glob.glob(pipes_wildcard):
        subprocess.Popen(f"rm {pipes_wildcard}", shell=True).communicate()
