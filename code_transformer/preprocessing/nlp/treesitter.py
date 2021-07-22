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

logger = get_logger(__file__)

if not exists(BUILD_PATH):
    mkdir(BUILD_PATH)

languages = [
    "python",
    "javascript",
    "go",
    "cpp"
]

for lang in languages:
    ts_lang_repo = f"tree-sitter-{lang}"
    if not exists(ts_lang_repo):
        call = f"git clone https://github.com/tree-sitter/{ts_lang_repo} {join(BUILD_PATH, ts_lang_repo)}"
        cabal_call = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True, shell=True)
        _ = cabal_call.communicate()
        cabal_call.wait()

Language.build_library(
  "build/my-languages.so",
  [
    "build/tree-sitter-go",
    "build/tree-sitter-javascript",
    "build/tree-sitter-python",
    "build/tree-sitter-cpp"
  ]
)

GO_LANGUAGE = Language('build/my-languages.so', 'go')
JS_LANGUAGE = Language('build/my-languages.so', 'javascript')
PY_LANGUAGE = Language('build/my-languages.so', 'python')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')


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


def cpp_to_ast(*code_snippets):
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)

    asts = []
    idx_successful = []
    for i, code_snippet in enumerate(code_snippets):
        tree = parser.parse(bytes(code_snippet, "utf8"))
        idx2node = {i: node for i, node in enumerate(traverse_tree(tree=tree))}
        node2idx = {(node.start_point, node.end_point): i for i, node in idx2node.items()}

        ast = dict(language="cpp", path="")
        ast["vertices"] = [
            {
                "span": {
                    "end": {
                        "column": int(node.end_point[0]),
                        "line": int(node.end_point[1])
                    },
                    "start": {
                        "column": int(node.start_point[0]),
                        "line": int(node.start_point[1])
                    },
                },
                "term": node.type,
                "vertexId": i
            }
            for i, node in idx2node.items()]
        ast["edges"] = [
            {
                "source": node2idx[(node.parent.start_point, node.parent.end_point)],
                "target": i
            }
            for i, node in idx2node.items() if node.parent is not None
        ]

        asts.append(ast)
        idx_successful.append(i)
    return {"files": asts}, idx_successful
