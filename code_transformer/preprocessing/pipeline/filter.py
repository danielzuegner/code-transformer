"""
A Collection of low-level textual code snippet filters used during stage 1 preprocessing.
Implements masking of numbers and strings, removing comments, detecting indentations and sub-tokenization.
"""

import io

import pygments.token
from collections import Counter

from code_transformer.preprocessing.nlp.text import RangeInterval, TextPosition
from code_transformer.preprocessing.nlp.tokenization import split_identifier_into_parts, CTToken, Token

from abc import ABC, abstractmethod

from typing import List


class CodeFilter(ABC):

    @abstractmethod
    def filter(self, processed_code: str):
        pass


class TokenFilter(ABC):

    def set_processed_code(self, processed_code: str):
        self.processed_code = processed_code

    @abstractmethod
    def filter(self, tokens: List[Token]):
        pass


class CodePreprocessor:

    def __init__(self, tokenizer, code_filters: List[CodeFilter], token_filters: List[TokenFilter]):
        self.tokenizer = tokenizer
        self.code_filters = code_filters
        self.token_filters = token_filters

    def process(self, code):
        processed_code = code
        for code_filter in self.code_filters:
            processed_code = code_filter.filter(processed_code)
        tokens = self.tokenizer.tokenize(processed_code)
        for token_filter in self.token_filters:
            token_filter.set_processed_code(processed_code)
            tokens = token_filter.filter(tokens)
        return processed_code, list(tokens)


class CodePreprocessingException(Exception):

    def __init__(self, code_snippet, msg=""):
        self.code_snippet = code_snippet
        self.msg = msg

    def __str__(self):
        return f"Error while processing snippet {self.code_snippet}: \n {self.msg}"


# =============================================================================
# Stage 1 CodeFilter: Directly transform code snippet text
# =============================================================================

class CommentsRemover(CodeFilter):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def filter(self, code):
        tokens = self.tokenizer.tokenize(code)
        last_token_comment = None
        for t in tokens:
            if last_token_comment:
                if t.string == '\n':
                    last_token_comment.string = ''
                else:
                    last_token_comment.string = '\n'
                last_token_comment = None
            if t.token_type in pygments.token.Comment.Multiline:
                t.string = ' '
            elif t.token_type in pygments.token.Literal.String.Doc:
                t.string = ' '
            elif t.token_type in pygments.token.Comment:
                last_token_comment = t
        return ''.join([t.string for t in tokens])


class EmptyLinesRemover(CodeFilter):

    def filter(self, code):
        lines = io.StringIO(code).readlines()
        stripped_lines = []
        last_line_empty = False
        for line in lines:
            if last_line_empty:
                if not line.isspace():
                    last_line_empty = False
                    stripped_lines.append(line)
            else:
                if line.isspace():
                    last_line_empty = True
                stripped_lines.append(line)
        return ''.join(stripped_lines)


# =============================================================================
# Stage 2 TokenFilter: transform already generated tokens
# =============================================================================

class StringMasker(TokenFilter):

    def __init__(self, string_mask):
        self.string_mask = string_mask

    def filter(self, tokens):
        string_start = None
        string_end = None
        for t in tokens:
            # Replace String literals with a MASK
            if t.token_type not in pygments.token.Literal.String and string_start:
                string_token = CTToken(self.string_mask, RangeInterval(string_start, string_end),
                                       pygments.token.Literal.String)
                string_start = None
                yield string_token

            if t.token_type in pygments.token.Literal.String:
                if not string_start:
                    string_start = t.source_span.start_pos
                    string_end = t.source_span.end_pos
                else:
                    string_end = t.source_span.end_pos
            else:
                yield t


class NumbersMasker(TokenFilter):

    def __init__(self, numbers_mask):
        self.numbers_mask = numbers_mask

    def filter(self, tokens):
        for t in tokens:
            if t.token_type in pygments.token.Literal.Number:
                t.string = self.numbers_mask
            yield t


class IndentTransformer(TokenFilter):

    def __init__(self, indent_token, dedent_token, fix_first_indent=True, allow_empty_methods=False):
        self.indent_token = indent_token
        self.dedent_token = dedent_token
        self.fix_first_indent = fix_first_indent
        self.allow_empty_methods = allow_empty_methods

    def filter(self, tokens):
        lines = self.processed_code.splitlines()

        # Get indentations from lines
        indents = []
        for line in lines:
            indent = abs(len(line) - len(line.lstrip()))
            if not line.isspace():
                indents.append(indent)

        # Use indent changes between two consecutive lines as heuristic to detect the indentation style of a snippet
        indent_changes = [abs(i1 - i2) for i1, i2 in zip(indents[:-1], indents[1:])]
        indent_changes = Counter([ic for ic in indent_changes if ic > 0])
        most_common = 0
        most_common_indent_changes = []
        for indent_change, frequency in indent_changes.items():
            if frequency > most_common:
                most_common_indent_changes = []
                most_common = frequency
            if frequency >= most_common:
                most_common_indent_changes.append(indent_change)

        # if there are multiple most common indent changes (e.g., 2 and 4) then we use the smaller one as indent style
        if not most_common_indent_changes:
            if self.allow_empty_methods:
                # When the method does not contain a real body, the indentation style cannot be detected. If
                # allow_empty_methods is set, then the IndentTransformer just passes through all tokens
                for t in tokens:
                    if t.string.isspace():
                        if '\n' in t.string:
                            t.string = '\n'
                            yield t
                    else:
                        yield t
                return
            else:
                raise CodePreprocessingException('',
                                                 f"Snippet `{' '.join([t.string for t in tokens])}` is not comprised of more than one line")
        indent_style = min(most_common_indent_changes)

        # Go through all tokens, identify new lines and remove whitespaces at the beginning of new lines. Indent/Dedent
        # tokens will be inserted when indentation changes accordingly
        beginning_of_line = True
        previous_indent = 0
        current_indent = 0
        first_indentation = True
        for t in tokens:
            if t.string.isspace():
                # end of line: have to reset variables for next line
                if '\n' in t.string:
                    lines = t.string.splitlines()

                    beginning_of_line = True
                    # it can happen that multiple \n are in one token. Additionally, there are sometimes whitespaces
                    # after a \n in the same token. If these whitespaces appear after the last \n in a token they
                    # must be added to the indentation of the new line.
                    current_indent = len(lines[-1])
                    lines[-1] = ''
                    if len(lines) > 1:
                        t.string = '\n'.join(lines)
                    else:
                        t.string = '\n'
                    yield t
                # If whitespaces appear at beginning of line we just sum their length
                elif beginning_of_line:
                    current_indent += len(t.string)

            # Once we encounter the first token in a line that is not a whitespace anymore, we can generate
            # indent/dedent tokens
            elif beginning_of_line:
                indent_change = current_indent - previous_indent
                # Only generate tokens if the indent change fits the indent style
                if indent_change % indent_style == 0:
                    indent_dedent_token = self.indent_token if indent_change > 0 else self.dedent_token

                    # Sometimes, the code is formatted in a way such that there are multiple indents in the first
                    # line of the function. Here we fix it to only generate a single indent token in the beginning
                    if first_indentation and self.fix_first_indent and t.source_span.start_pos.line > 1:
                        indent_change = indent_style
                        first_indentation = False
                    # Generate a separate token for every indent/dedent
                    if indent_change > 0:
                        for i in range(int(indent_change / indent_style)):
                            yield CTToken(self.indent_token,
                                          RangeInterval(
                                               TextPosition(t.source_span.start_pos.line, i * indent_style + 1),
                                               TextPosition(t.source_span.start_pos.line,
                                                            (i + 1) * indent_style + 1)),
                                          self.indent_token)
                    else:
                        for i in range(-int(indent_change / indent_style)):
                            yield CTToken(self.dedent_token,
                                          RangeInterval(TextPosition(t.source_span.start_pos.line, 1),
                                                         TextPosition(t.source_span.start_pos.line, 1)),
                                          self.dedent_token)

                    previous_indent = current_indent
                beginning_of_line = False
                yield t
            else:
                yield t


class WhitespaceRemover(TokenFilter):

    def filter(self, tokens):
        for t in tokens:
            if t.string != ' ':
                yield t


def pad_or_truncate(seq, length, padding):
    padded_seq = seq.copy()
    if not isinstance(padded_seq, list):
        padded_seq = [padded_seq]
    if len(padded_seq) > length:
        return padded_seq[:length]
    else:
        pad_size = length - len(padded_seq)
        padded_seq.extend([padding for i in range(pad_size)])
        return padded_seq


class SubTokenizer(TokenFilter):
    """
    The final TokenFilter that transforms all tokens into CSNTokens
    """

    def __init__(self, num_sub_tokens):
        self.num_sub_tokens = num_sub_tokens

    def filter(self, tokens: List[Token]):
        for t in tokens:
            if t.token_type in pygments.token.Name:
                t.string = split_identifier_into_parts(t.string)
            else:
                t.string = [t.string]
            t.string = t.string[:self.num_sub_tokens]
            yield CTToken(t.string, t.source_span, t.token_type)


# =============================================================================
# Stage 3: Filters that filter entire samples
# =============================================================================

class TokensLimiter:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, tokens_batch):
        samples_filter = []
        for tokens in tokens_batch:
            if tokens is None:
                # This happens when the snippet was marked as problematic in an earlier stage. In this case, the sample
                # is abandoned
                samples_filter.append(False)
            else:
                if self.threshold is None:
                    accept = True
                else:
                    accept = len(tokens) <= self.threshold
                # Sometimes it can happen that the Pygments tokenizer produces an error Token. In this case, we
                # abandon the whole sample
                accept &= pygments.token.Token.Error not in {token.token_type for token in tokens}
                samples_filter.append(accept)
        return samples_filter
