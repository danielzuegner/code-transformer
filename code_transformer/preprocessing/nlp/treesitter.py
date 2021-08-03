"""
Obtain an AST from a CPP method snippet.
"""

import json
import subprocess

from code_transformer.env import JAVA_EXECUTABLE, JAVA_PARSER_EXECUTABLE
from code_transformer.utils.log import get_logger
from tree_sitter import Language, Parser, Tree
from os.path import exists, join
from os import mkdir

BUILD_PATH = "build"
TREE_SITTER_BIN = join(BUILD_PATH, "code-transformer-languages.so")

logger = get_logger(__file__)

if not exists(BUILD_PATH):
    mkdir(BUILD_PATH)

languages = ["cpp"]

for lang in languages:
    ts_lang_repo = f"tree-sitter-{lang}"
    if not exists(ts_lang_repo):
        call = f"git clone https://github.com/tree-sitter/{ts_lang_repo} {join(BUILD_PATH, ts_lang_repo)}"
        cabal_call = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True, shell=True)
        _ = cabal_call.communicate()
        cabal_call.wait()

Language.build_library(TREE_SITTER_BIN, [join(BUILD_PATH, f"tree-sitter-{lang}") for lang in languages])


def traverse_tree(tree: Tree):
    cursor = tree.walk()

    reached_root = False
    while not reached_root:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


def treesitter_ast(language, *code_snippets):
    parser = Parser()
    parser.set_language(Language(TREE_SITTER_BIN, language))

    asts = []
    idx_successful = []
    for idx, code_snippet in enumerate(code_snippets):
        tree = parser.parse(bytes(code_snippet, "utf8"))
        idx2node = {i: node for i, node in enumerate(traverse_tree(tree=tree)) if node.parent is not None}
        node2idx = {(node.start_point, node.end_point): i for i, node in idx2node.items()}

        ast = dict(language="cpp", path="")
        ast["vertices"] = [
            {
                "span": {
                    "end": {
                        "column": node.end_point[1] + 1,
                        "line": node.end_point[0] + 1
                    },
                    "start": {
                        "column": node.start_point[1] + 1,
                        "line": node.start_point[0] + 1
                    },
                },
                "term": node.type,
                "vertexId": i
            }
            for i, node in idx2node.items()]

        def get_range(node):
            return node.start_point, node.end_point

        ast["edges"] = []
        for i, node in idx2node.items():
            rng = get_range(node.parent)
            if (rng in node2idx) and (i != node2idx[rng]):
                ast["edges"].append({
                    "source": node2idx[get_range(node.parent)],
                    "target": i
                })

        asts.append(ast)
        idx_successful.append(idx)
    return {"files": asts}, idx_successful
