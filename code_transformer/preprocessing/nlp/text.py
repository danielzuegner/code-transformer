import functools
import io


@functools.total_ordering
class TextPosition:
    """
    By using total_ordering, only __lt__ and __eq__ have to be defined in order to make objects of type TextPosition
    comparable.
    Lines and columns start counting with 1.
    """

    def __init__(self, line, column):
        assert line > 0, f"Text lines must be positive numbers, not `{line}`!"
        assert column > 0, f"Text columns must be positive numbers, not `{column}`!"
        self.line = line
        self.column = column

    @staticmethod
    def from_ast_pos(ast_pos):
        return TextPosition(ast_pos['line'], ast_pos['column'])

    def __lt__(self, other):
        if self.line == other.line:
            return self.column < other.column
        return self.line < other.line

    def __eq__(self, other):
        return self.line == other.line and self.column == other.column

    def __str__(self):
        return f"{self.line},{self.column}"


class RangeInterval:
    """
    Right-open integer interval representing spans in text.
    """

    def __init__(self, start_pos, end_pos):
        assert end_pos >= start_pos, f"An interval cannot have a negative range! {start_pos} - {end_pos}"
        self.start_pos = start_pos
        self.end_pos = end_pos

    @staticmethod
    def from_semantic(span):
        return RangeInterval(TextPosition.from_ast_pos(span['start']), TextPosition.from_ast_pos(span['end']))

    @staticmethod
    def from_java_parser(span):
        return RangeInterval(TextPosition.from_ast_pos(span['begin']), TextPosition.from_ast_pos(span['end']))

    @staticmethod
    def from_python_token(span):
        start_line, start_col = span.start
        end_line, end_col = span.end
        return RangeInterval(TextPosition(start_line, start_col + 1), TextPosition(end_line, end_col + 1))

    @staticmethod
    def from_compressed(compressed_interval):
        return RangeInterval(TextPosition(compressed_interval[0][0], compressed_interval[0][1]),
                             TextPosition(compressed_interval[1][0], compressed_interval[1][1]))

    @staticmethod
    def empty_interval():
        return RangeInterval(TextPosition(1, 1), TextPosition(1, 1))

    def compress(self):
        return ((self.start_pos.line, self.start_pos.column), (self.end_pos.line, self.end_pos.column))

    def contains(self, other):
        return self.start_pos <= other.start_pos and self.end_pos >= other.end_pos

    def intersects(self, other):
        return self.start_pos < other.end_pos and self.end_pos > other.start_pos

    def substring(self, text):
        lines = io.StringIO(text).readlines()
        lines = [line for i, line in enumerate(lines) if self.start_pos.line <= i + 1 <= self.end_pos.line]
        substring = []
        for i, line in enumerate(lines):
            idx_start = 0
            idx_end = len(line)
            if i == 0:
                idx_start = self.start_pos.column - 1
            if i == len(lines) - 1:
                idx_end = self.end_pos.column - 1
            substring.append(line[idx_start:idx_end])

        return ''.join(substring)

    def is_smaller_than(self, other):
        line_span_self = self.end_pos.line - self.start_pos.line
        line_span_other = other.end_pos.line - other.start_pos.line
        if line_span_self < line_span_other:
            return True
        elif line_span_self > line_span_other:
            return False
        else:
            col_span_self = self.end_pos.column - self.start_pos.column
            col_span_other = other.end_pos.column - other.start_pos.column
            return col_span_self - col_span_other < 0

    def __str__(self):
        return f"[{self.start_pos} - {self.end_pos}]"
