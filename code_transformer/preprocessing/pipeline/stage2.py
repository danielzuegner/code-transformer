"""
Defines the sample classes that will hold the result of stage2 preprocessing.
The actual preprocessing is done in graph/distances.py and graph/transform.py
"""

from typing import List

from code_transformer.preprocessing.nlp.tokenization import CTToken
from code_transformer.utils.data import tensor_to_tuple, tuple_to_tensor


class CTStage2Sample:

    def __init__(self, tokens: List[CTToken], graph_sample: dict, token_mapping: dict, stripped_code_snippet: str,
                 func_name: str, docstring: str, encoded_func_name: List = None):
        self.tokens = tokens
        self.graph_sample = graph_sample
        self.token_mapping = token_mapping
        self.stripped_code_snippet = stripped_code_snippet
        self.func_name = func_name
        self.docstring = docstring
        self.encoded_func_name = encoded_func_name

    @staticmethod
    def from_compressed(compressed_sample):
        graph_sample = compressed_sample.graph_sample
        for i, distance in enumerate(graph_sample['distances']):
            if isinstance(distance[0], tuple):
                graph_sample['distances'][i] = (tuple_to_tensor(distance[0]).to_dense(), distance[1], distance[2])
        return CTStage2Sample(compressed_sample.tokens, graph_sample, compressed_sample.token_mapping,
                              compressed_sample.stripped_code_snippet, compressed_sample.func_name,
                              compressed_sample.docstring,
                              compressed_sample.encoded_func_name if hasattr(compressed_sample,
                                                                              'encoded_func_name') else None)

    def compress(self):
        for i, distance in enumerate(self.graph_sample['distances']):
            # shortest paths distance matrix is dense, thus it is excluded
            if not distance[2] == 'shortest_paths':
                self.graph_sample['distances'][i] = (tensor_to_tuple(distance[0]), distance[1], distance[2])


class CTStage2MultiLanguageSample(CTStage2Sample):

    def __init__(self, tokens: List[CTToken], graph_sample: dict, token_mapping: dict, stripped_code_snippet: str,
                 func_name: str, docstring: str, language: str, encoded_func_name: List = None):
        super().__init__(tokens, graph_sample, token_mapping, stripped_code_snippet, func_name, docstring,
                         encoded_func_name)
        self.language = language

    @staticmethod
    def from_compressed(compressed_sample):
        graph_sample = compressed_sample.graph_sample
        for i, distance in enumerate(graph_sample['distances']):
            if isinstance(distance[0], tuple):
                graph_sample['distances'][i] = (tuple_to_tensor(distance[0]).to_dense(), distance[1], distance[2])
        return CTStage2MultiLanguageSample(compressed_sample.tokens, graph_sample, compressed_sample.token_mapping,
                                           compressed_sample.stripped_code_snippet, compressed_sample.func_name,
                                           compressed_sample.docstring, compressed_sample.language,
                                           compressed_sample.encoded_func_name if hasattr(compressed_sample,
                                                                                           'encoded_func_name') else None)
