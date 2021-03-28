"""
Uses the JavaParser .jar to obtain an AST from a Java method snippet.
"""

import json
import subprocess

from env import JAVA_EXECUTABLE, JAVA_PARSER_EXECUTABLE
from code_transformer.utils.log import get_logger

JAVA_PARSER_CMD = f"{JAVA_EXECUTABLE} -jar {JAVA_PARSER_EXECUTABLE}"

logger = get_logger(__file__)


def java_to_ast(*code_snippets):
    asts = []
    idx_successful = []
    for i, code_snippet in enumerate(code_snippets):
        java_parser_call = subprocess.Popen(JAVA_PARSER_CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True, shell=True)
        output, errors = java_parser_call.communicate(code_snippet)
        java_parser_call.wait()
        if not errors == "":
            logger.warn(errors)
            logger.warn(code_snippet)
        else:
            output = json.loads(output)
            asts.append(output)
            idx_successful.append(i)
    return asts, idx_successful
