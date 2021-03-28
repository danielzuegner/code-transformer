import io
import tokenize
from abc import ABC
from functools import lru_cache
from typing import List

import pygments.lexers

from code_transformer.modeling.constants import UNKNOWN_TOKEN, NUM_SUB_TOKENS_METHOD_NAME
from code_transformer.preprocessing.nlp.semantic import semantic_parse
from code_transformer.preprocessing.nlp.text import TextPosition, RangeInterval


class Tokenizer(ABC):
    def tokenize(self, text):
        pass


class DefaultTokenizer(Tokenizer):

    def tokenize(self, text):
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        return [Token.from_python_token(token) for token in tokens]


class SemanticTokenizer(Tokenizer):

    def __init__(self, language):
        self.language = language

    def tokenize(self, text):
        result = semantic_parse(self.language, "--symbols", text)
        return [Token.from_semantic(token) for token in result['files'][0]['symbols']]


class PygmentsTokenizer(Tokenizer):

    def __init__(self, language):
        self.language = language
        self._lexer = pygments.lexers.get_lexer_by_name(language)

    def tokenize(self, code):
        lines = io.StringIO(code).readlines()
        line_lengths = {i + 1: len(line) for i, line in enumerate(lines)}

        # When using multiprocessing with joblib it sometimes can happen that the lexer is not properly instantiated
        # In this case, we just create a new object
        if not hasattr(self._lexer, "_tokens"):
            self._lexer = pygments.lexers.get_lexer_by_name(self.language)
        unprocessed_tokens = self._lexer.get_tokens_unprocessed(code)

        current_line = 1
        current_line_start_pos = 0
        tokens = []
        # Calculate source spans
        for pos, token_type, string in unprocessed_tokens:
            if current_line_start_pos + line_lengths[current_line] <= pos:
                current_line_start_pos += line_lengths[current_line]
                current_line += 1
            start_line = current_line
            start_col = pos - current_line_start_pos + 1
            end_line = current_line
            multiline_pos = current_line_start_pos
            while multiline_pos + line_lengths[end_line] < pos + len(string):
                multiline_pos += line_lengths[end_line]
                end_line += 1
            current_line = end_line
            current_line_start_pos = multiline_pos
            end_col = pos - multiline_pos + len(string) + 1
            source_span = RangeInterval(TextPosition(start_line, start_col), TextPosition(end_line, end_col))
            tokens.append(Token(string, source_span, token_type))

        return tokens


class Token:

    def __init__(self, string, source_span, token_type):
        self.string = string
        self.source_span = source_span
        self.token_type = token_type

    @staticmethod
    def from_semantic(token):
        return Token(token['symbol'], RangeInterval.from_semantic(token['span']), token['kind'])

    @staticmethod
    def from_python_token(token):
        return Token(token.string, RangeInterval.from_python_token(token), token.type)

    @staticmethod
    def from_compressed(compressed_token):
        return Token(compressed_token[0], RangeInterval.from_compressed(compressed_token[1]), compressed_token[2])

    def compress(self):
        return (self.string, self.source_span.compress(), self.token_type)

    def __str__(self):
        return f"{self.source_span} ({self.token_type}) '{self.string}'"


class CTToken(Token):

    def __init__(self, sub_tokens, source_span, token_type):
        super().__init__(sub_tokens, source_span, token_type)
        self.sub_tokens = self.string
        self.original_sub_tokens = self.string.copy() if isinstance(self.string, list) else self.string

    @staticmethod
    def from_compressed(compressed_token):
        return CTToken(compressed_token[0], RangeInterval.from_compressed(compressed_token[1]), compressed_token[2])


def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    """
    if not len(camel_case_identifier):
        return []

    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    snake_case = identifier.split("_")

    identifier_parts = []  # type: List[str]
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(s.lower() for s in split_camelcase(part))
    if len(identifier_parts) == 0:
        return [identifier]
    return identifier_parts


def method_name_to_tokens(method_name: str) -> List[str]:
    func_name = method_name[method_name.rindex('.') + 1:] if '.' in method_name else method_name
    label_tokens = split_identifier_into_parts(func_name)
    label_tokens = label_tokens[:NUM_SUB_TOKENS_METHOD_NAME]
    return label_tokens


def get_idx_no_punctuation(decoded_tokens):
    return [i for i, token in enumerate(decoded_tokens)
            if len(token) > 1  # tokens with 2+ parts
            or (any(c.isalpha() for c in token[0]) and not token[0] in {'[INDENT]',
                                                                        '[DEDENT]'})  # non-indent tokens that are words
            or token[0] == '_'  # Sometimes function names/variables can be just an underscore
            or token[0] == UNKNOWN_TOKEN  # unknown token could have been anything, including an identifier
            ]
