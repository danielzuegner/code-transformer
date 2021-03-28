"""
Implements the stage1 preprocessing pipeline which consists of several steps:
 (1) Tokenizing the code snippet
 (2) Apply snippet-level filtering (removing comments, masking strings, ...)
 (3) Parsing the AST from the code snippet (this takes the biggest amount of time)
 (4) Transform the AST parsing result into a graph
 (5) Calculate mapping between tokens and graph nodes
"""

from itertools import compress
from typing import List

from code_transformer.modeling.constants import NUM_SUB_TOKENS, MAX_NUM_TOKENS, MASK_STRING, MASK_NUMBER, \
    INDENT_TOKEN, DEDENT_TOKEN
from code_transformer.preprocessing.graph.ast import ASTGraph
from code_transformer.preprocessing.nlp.javaparser import java_to_ast
from code_transformer.preprocessing.nlp.semantic import semantic_parse
from code_transformer.preprocessing.nlp.tokenization import PygmentsTokenizer, CTToken
from code_transformer.preprocessing.pipeline.filter import CodePreprocessor, CommentsRemover, \
    EmptyLinesRemover, StringMasker, NumbersMasker, IndentTransformer, SubTokenizer, WhitespaceRemover, TokensLimiter, \
    CodePreprocessingException
from code_transformer.utils.log import get_logger

logger = get_logger(__name__)


class CTStage1Preprocessor:

    def __init__(self, language, allow_empty_methods=False, use_tokens_limiter=True, max_num_tokens=MAX_NUM_TOKENS):
        self.language = language
        tokenizer = PygmentsTokenizer(language)

        comments_remover = CommentsRemover(tokenizer)
        empty_lines_remover = EmptyLinesRemover()

        string_masker = StringMasker(MASK_STRING)
        numbers_masker = NumbersMasker(MASK_NUMBER)

        indent_transformer = IndentTransformer(INDENT_TOKEN, DEDENT_TOKEN, fix_first_indent=True,
                                               allow_empty_methods=allow_empty_methods)
        sub_tokenizer = SubTokenizer(NUM_SUB_TOKENS)
        whitespace_remover = WhitespaceRemover()

        self.code_preprocessor = CodePreprocessor(tokenizer, [comments_remover, empty_lines_remover],
                                                  [indent_transformer, whitespace_remover, string_masker,
                                                   numbers_masker, sub_tokenizer])
        self.tokens_limiter = TokensLimiter(max_num_tokens if use_tokens_limiter else None)

    def process(self, batch, process_identifier):
        func_names, docstrings, code_snippets = zip(*batch)

        # Step 1: generate tokens from code snippets and apply filtering on token level
        stripped_code_snippets = []
        tokens_batch = []
        for code in code_snippets:
            # It can happen that an exception occurs during filtering one of the snippets. In this case, it will
            # just be ignored
            try:
                stripped_code_snippet, tokens = self.code_preprocessor.process(code)
                stripped_code_snippets.append(stripped_code_snippet)
                tokens_batch.append(tokens)
            except CodePreprocessingException as e:
                logger.warning(str(e))
                tokens_batch.append(None)  # Signalize that this sample is to be ignored
                stripped_code_snippets.append(None)

        # Step 2: Filtering on snippet level
        samples_filter = self.tokens_limiter(tokens_batch)
        if all([not accepted for accepted in samples_filter]):
            raise PreprocessingException(process_identifier,
                                         "No snippets left after filtering step. Skipping batch")

        assert all([len(l) == len(code_snippets) for l in [func_names, docstrings, stripped_code_snippets,
                                                           tokens_batch]]), "Intermediate lists differ in length"
        # Select samples that passed the tokens limiter filter
        func_names, docstrings, stripped_code_snippets, tokens_batch = zip(*compress(zip(func_names, docstrings,
                                                                                         stripped_code_snippets,
                                                                                         tokens_batch),
                                                                                     samples_filter))

        # Step 3: Create ASTs from code snippets
        if self.language == 'java':
            ast_batch, idx_successful = java_to_ast(*stripped_code_snippets)
        else:
            ast_batch, idx_successful = semantic_parse(self.language, "--fail-on-parse-error", "--json-graph",
                                                       process_identifier, *stripped_code_snippets, quiet=False)

        # In case, some of the snippets in the batch could not be parsed, we just ignore them
        if idx_successful is not None:
            stripped_code_snippets = [stripped_code_snippets[i] for i in idx_successful]
            tokens_batch = [tokens_batch[i] for i in idx_successful]
            func_names = [func_names[i] for i in idx_successful]
            docstrings = [docstrings[i] for i in idx_successful]

        # Step 4: Transform AST into graph and prune
        ast_graph_batch = []
        if self.language == 'java':
            for ast_graph in ast_batch:
                ast = ASTGraph.from_java_parser(ast_graph)
                ast.prune()
                ast_graph_batch.append(ast)
        else:
            for ast_graph in ast_batch['files']:
                ast = ASTGraph.from_semantic(ast_graph)
                # Prune 'Empty' nodes to save some space later when calculating NxN matrices on ast
                ast.prune()
                ast_graph_batch.append(ast)

        # Step 5: Calculate mapping between tokens and graph nodes
        token_mapping_batch = []
        for tokens, ast_graph in zip(tokens_batch, ast_graph_batch):
            token_mapping = {i: ast_graph.find_smallest_encompassing_interval(token.source_span) for i, token in
                             enumerate(tokens)}
            token_mapping_batch.append(token_mapping)

        # Step 6: Put everything together
        samples_batch = []

        batch_parts = [tokens_batch, ast_graph_batch, token_mapping_batch, stripped_code_snippets, func_names,
                       docstrings]
        num_samples = len(batch_parts[0])
        assert all([len(batch_part) == num_samples for batch_part in batch_parts]), "Batch parts do not have same " \
                                                                                    "length"
        for sample in zip(*batch_parts):
            s = CTStage1Sample(*sample)
            samples_batch.append(s)

        return samples_batch


class CTStage1Sample:

    def __init__(self, tokens: List[CTToken], ast, token_mapping, stripped_code_snippet, func_name, docstring):
        self.tokens = tokens
        self.ast = ast
        self.token_mapping = token_mapping
        self.stripped_code_snippet = stripped_code_snippet
        self.func_name = func_name
        self.docstring = docstring
        self.encoded_func_name = None

    def __iter__(self):
        """
        Allows unpacking a CSNSample like a tuple
        """
        return iter((self.tokens, self.ast, self.token_mapping, self.stripped_code_snippet))

    def compress(self):
        # In stage 2, ast is only a dict. No need to compress
        if isinstance(self.ast, ASTGraph):
            self.ast = self.ast.compress()
        self.tokens = [token.compress() for token in self.tokens]

        return self.tokens, self.ast, self.token_mapping, self.stripped_code_snippet, self.func_name, self.docstring

    @staticmethod
    def from_compressed(sample):
        # Cannot be called on the same sample twice, as the original sample is mutated and reused
        if isinstance(sample, CTStage1Sample):
            return sample

        tokens, ast_graph, token_mapping, stripped_code_snippet, func_name, docstring = sample
        tokens = [CTToken.from_compressed(token) for token in tokens]
        ast_graph = ASTGraph.from_compressed(ast_graph)

        return CTStage1Sample(tokens, ast_graph, token_mapping, stripped_code_snippet, func_name, docstring)

    def reconstruct_snippet(self):
        output = []
        for token in self.tokens:
            output.append(str(token.string))
        return ' '.join(output)

    def remove_punctuation(self):
        # Calculate indices of tokens that should be kept, i.e., are tokens like identifiers or types
        idx = [i for i, token in enumerate(self.tokens) if len(token.sub_tokens) > 1
               or (any([c.isalpha() for c in token.sub_tokens[0]]) and not token.sub_tokens[0] in {'[INDENT]',
                                                                                                   '[DEDENT]'})]

        self.tokens = [self.tokens[i] for i in idx]
        self.token_mapping = {t: n for t, n in self.token_mapping.items() if t in idx}


class PreprocessingException(Exception):
    def __init__(self, batch_id, msg):
        self.batch_id = batch_id
        self.msg = msg

    def __str__(self):
        return f"Error processing batch {self.batch_id}: {self.msg}"
