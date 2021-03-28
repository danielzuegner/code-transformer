"""
This is an adaptation of code2seq AST path walks for CSN datasets and languages.
It is used in the preprocessing for code2seq. The entry point is the function __collect_samples().
More programming languages can be supported by adding a configuration to METHOD_NODE_TYPE below.
"""

import itertools
import json
import re

import joblib
import tqdm

from code_transformer.preprocessing.graph.ast import ASTGraph

# =========================================================================
# Settings
# =========================================================================


METHOD_NAME, NUM = 'METHODNAME', 'NUM'

# Configuration for masking the method name from sampled AST paths
# First tuple entry: language-specific types of the node corresponding to the method in the AST
# Second tuple entry: language-specific type of the node containing the method name in the AST
METHOD_NODE_TYPE = {
    'python': ({'Function', 'Method'}, 'Identifier'),
    'go': ({'Function', 'Method'}, 'Identifier'),
    'ruby': ({'Function', 'Method'}, 'Identifier'),
    'javascript': ({'Function', 'Method'}, 'Identifier'),
    'java-small': ({'MethodDeclaration'}, 'SimpleName')
}


# =========================================================================
# END Settings
# =========================================================================


def is_identifier(language, node_type):
    if language in {'python', 'go', 'ruby', 'javascript'}:
        return node_type in {'Identifier', 'Boolean', 'Null'}
    elif language == 'java-small':
        return "Name" in node_type or node_type in {'ThisExpr', 'SuperExpr',
                                                    'BooleanLiteralExpr'}
    else:
        raise NotImplementedError(f'code2seq not implemented for language {language}')


def is_num(language, node_type):
    if language in {'python', 'go', 'ruby', 'javascript'}:
        return node_type in {'Integer', 'Float'}
    elif language == 'java-small':
        return node_type in {'IntegerLiteralExpr', 'DoubleLiteralExpr', 'LongLiteralExpr'}
    else:
        raise NotImplementedError(f'code2seq not implemented for language {language}')


def __collect_asts(json_file):
    asts = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            ast = json.loads(line.strip())
            asts.append(ast)

    return asts


def __terminals(ast, node_index, args, language, function_name_idx):
    stack, paths = [], []

    def dfs(v):
        stack.append(v)

        v_node = ast[v]

        if hasattr(v_node, 'value'):
            if v == function_name_idx:  # Top-level func def node.
                if args.use_method_name:
                    paths.append((stack.copy(), METHOD_NAME))
            else:
                v_type = v_node.node_type

                if is_identifier(language, v_type):
                    if not re.match("^[a-zA-Z0-9_?$@!*`&~:/'.]+$", v_node.value):
                        raise ValueError(f'Identifier `{v_node.value}` contains illegal characters. Cannot be used for '
                                         'code2seq. Abandoning sample. ')

                    paths.append((stack.copy(), v_node.value))
                elif args.use_nums and is_num(language, v_type):
                    paths.append((stack.copy(), NUM))
                else:
                    pass

        for child in v_node.children:
            dfs(child)

        stack.pop()

    dfs(node_index)

    return paths


def __merge_terminals2_paths(v_path, u_path):
    s, n, m = 0, len(v_path), len(u_path)
    while s < min(n, m) and v_path[s] == u_path[s]:
        s += 1

    prefix = list(reversed(v_path[s:]))
    lca = v_path[s - 1]
    suffix = u_path[s:]

    return prefix, lca, suffix


def __raw_tree_paths(ast, node_index, args, language, function_name_idx):
    tnodes = __terminals(ast, node_index, args, language, function_name_idx)

    tree_paths = []
    for (v_path, v_value), (u_path, u_value) in itertools.combinations(
            iterable=tnodes,
            r=2,
    ):
        prefix, lca, suffix = __merge_terminals2_paths(v_path, u_path)
        if (len(prefix) + 1 + len(suffix) <= args.max_path_length) \
                and (abs(len(prefix) - len(suffix)) <= args.max_path_width):
            path = prefix + [lca] + suffix
            tree_path = v_value, path, u_value
            tree_paths.append(tree_path)

    return tree_paths


def __delim_name(name):
    if name in {METHOD_NAME, NUM}:
        return name

    def camel_case_split(identifier):
        matches = re.finditer(
            '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split('_'):
        blocks.extend(camel_case_split(underscore_block))

    return '|'.join(block.lower() for block in blocks)


def __collect_sample(ast, fd_index, args, language, function_name_idx=None):
    root = ast[fd_index]
    if root.node_type not in METHOD_NODE_TYPE[language][0]:
        raise ValueError(f'Wrong node type. Got {root.node_type}')

    if function_name_idx is None:
        function_name_idx = fd_index

    if function_name_idx == -1:
        # Empty method names in JavaScript
        target = ''
    else:
        target = ast[function_name_idx].value

    tree_paths = __raw_tree_paths(ast, fd_index, args, language, function_name_idx)
    contexts = []
    for tree_path in tree_paths:
        start, connector, finish = tree_path

        start, finish = __delim_name(start), __delim_name(finish)
        connector = '|'.join(ast[v].node_type for v in connector)

        context = f'{start},{connector},{finish}'
        contexts.append(context)

    if len(contexts) == 0:
        return None

    target = __delim_name(target)
    context = ' '.join(contexts)

    return f'{target} {context}'


def __collect_samples(ast: ASTGraph, args, language, func_name=None):
    samples = []
    # n_idx_function = [n_idx for n_idx, node in ast.nodes.items() if node.node_type == 'Function']
    # if len(n_idx_function) == 1:
    #     sample = __collect_sample(ast, n_idx_function[0], args)
    #     if sample is not None:
    #         samples.append(sample)

    method_node_type, method_name_node_type = METHOD_NODE_TYPE[language]

    try:
        for node_index, node in ast.nodes.items():
            if node.node_type in method_node_type:
                # In Python, first Identifier after function token holds the function name

                function_name_idx = node_index
                if func_name is not None:
                    if func_name == '':
                        # JavaScript can have empty method names. Currently, we ignore these
                        function_name_idx = -1
                        break
                    else:
                        # Search through nodes until the one containing the method name is found
                        while True:
                            if function_name_idx >= len(ast.nodes):
                                function_name_idx = None
                                break
                            if ast.nodes[function_name_idx].value == func_name \
                                    and ast.nodes[function_name_idx].node_type == method_name_node_type:
                                break
                            function_name_idx += 1
                else:
                    while True:
                        if function_name_idx >= len(ast.nodes):
                            function_name_idx = None
                            break
                        if ast.nodes[function_name_idx].node_type == method_name_node_type:
                            break
                        function_name_idx += 1
                if function_name_idx is None:
                    raise ValueError(f'Could not find function name {func_name} in sample. Abandoning sample')
                sample = __collect_sample(ast, node_index, args, language, function_name_idx=function_name_idx)
                if sample is not None:
                    samples.append(sample)
                    break
                else:
                    raise ValueError(f"No code2seq paths were generated for sample {func_name}")
    except ValueError as e:
        print(e)

    return samples


def __collect_all_and_save(asts, args, output_file):
    parallel = joblib.Parallel(n_jobs=args.n_jobs)
    func = joblib.delayed(__collect_samples)

    samples = parallel(func(ast, args) for ast in tqdm.tqdm(asts))
    samples = list(itertools.chain.from_iterable(samples))

    with open(output_file, 'w') as f:
        for line_index, line in enumerate(samples):
            f.write(line + ('' if line_index == len(samples) - 1 else '\n'))
