"""
Vocabularies map sub tokens to unique IDs. Vocabularies can be specified to only contain the n most common tokens
or to only contain tokens that appear at least n times.
"""

from code_transformer.modeling.constants import UNKNOWN_TOKEN
from code_transformer.preprocessing.pipeline.stage1 import CTStage1Sample
from code_transformer.preprocessing.nlp.tokenization import method_name_to_tokens


class WordCounter:

    def __init__(self):
        self.words = dict()

    def update(self, word):
        # Do not strip new lines \n from word
        word = str(word).strip(" \t\r").lower()
        assert not word == '', f"WordCounter received an empty string in word `{word}` which is not desired"
        if word not in self.words:
            self.words[word] = 1
        else:
            self.words[word] += 1

    def to_vocabulary(self, limit_most_common: int = None, min_frequency=None, special_symbols: dict = None):
        if special_symbols is None:
            special_symbols = dict()
        assert limit_most_common is None or limit_most_common >= len(
            special_symbols), "Cannot calculate most_common for less than the size of special_symbols"

        # Need to spare one extra space for the UNKNOWN_WORD token
        if UNKNOWN_TOKEN not in special_symbols and limit_most_common:
            limit_most_common -= 1

        # Count word occurrences ignoring special symbols as they will be added to the vocabulary anyways
        # Keep in mind that special symbols can be uppercase while all other vocabulary words are always lowercase
        word_counts = {word: count for word, count in self.words.items() if word not in {symbol.lower() for symbol in
                                                                                         special_symbols}}
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        if min_frequency is not None:
            sorted_words = filter(lambda x: x[1] >= min_frequency, sorted_words)
        most_common_words = [sorted_word[0] for sorted_word in sorted_words]
        if limit_most_common is not None:
            most_common_words = most_common_words[:limit_most_common - len(special_symbols)]

        return Vocabulary(most_common_words, special_symbols)


class Vocabulary:

    def __init__(self, words, special_symbols: dict = None):
        """
        :param special_symbols: the provided dictionary will form the basis of the vocabulary.
                                Words contained in special_symbols will always be part of the vocabulary no matter
                                how often they actually appeared. if UNKNOWN_TOKEN is not yet part of special_symbols,
                                it will be added
        """
        if special_symbols is not None:
            self.special_symbols = set(special_symbols.keys())
            self.vocabulary = special_symbols.copy()

            # the first IDs in the vocabulary are reserved for special symbols
            next_id = 0 if not self.vocabulary else max(self.vocabulary.values()) + 1
            if UNKNOWN_TOKEN not in special_symbols.keys():
                self.special_symbols.add(UNKNOWN_TOKEN)
                self.vocabulary[UNKNOWN_TOKEN] = next_id
                next_id += 1
        else:
            self.special_symbols = {UNKNOWN_TOKEN}
            self.vocabulary = {UNKNOWN_TOKEN: 0}
            next_id = 1

        for word in words:
            self.vocabulary[word] = next_id
            next_id += 1

        self.reverse_vocabulary = {id: word for word, id in self.vocabulary.items() if not word == UNKNOWN_TOKEN}

    def reverse_lookup(self, id):
        if id not in self.reverse_vocabulary:
            return UNKNOWN_TOKEN
        else:
            return self.reverse_vocabulary[id]

    def __getitem__(self, word):
        if word not in self.special_symbols:
            # Keep new lines \n in word. Special symbols won't be cast to lowercase
            word = str(word).strip(" \t\r").lower()

        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            return self.vocabulary[UNKNOWN_TOKEN]

    def __contains__(self, word):
        if word not in self.special_symbols:
            # Keep new lines \n in word. Special symbols won't be cast to lowercase
            word = str(word).strip(" \t\r").lower()

        return word in self.vocabulary

    def __len__(self):
        return len(self.vocabulary)

    def __str__(self):
        return str(self.vocabulary)


class VocabularyBuilder:
    """
    Used in stage 1 to aggregate word counts of CSNSamples in order to build a vocabulary later
    """

    def __init__(self, word_counter, token_type_counter, node_type_counter):
        self.word_counter = word_counter
        self.token_type_counter = token_type_counter
        self.node_type_counter = node_type_counter

    def __call__(self, sample: CTStage1Sample):
        for token in sample.tokens:
            assert all([isinstance(st, str) for st in
                        token.string]), f"Some sub tokens ({token.string}) do not have string values. Has this sample " \
                                        f"already been vocabularized?"
            try:
                for st in token.sub_tokens:
                    self.word_counter.update(st)
                self.token_type_counter.update(token.token_type)
            except Exception as e:
                print(str(e))
                print(f"Token: {token.string}")
                print(f"sub tokens: {token.sub_tokens}")
                print(f"sample: {sample.stripped_code_snippet}")

        for node in sample.ast.nodes.values():
            self.node_type_counter.update(node.node_type)


class CodeSummarizationVocabularyBuilder(VocabularyBuilder):
    """
    Vocabulary Builder extension that is specific for the Code Summarization task.
    Accepts an additional `word_counter_labels` argument that builds a separate vocabulary that only contains the label
    tokens, i.e., the method names. The idea behind having separate vocabularies for method body tokens and method name
    tokens is that only choosing among tokens that appeared in method names during training induces a strong bias
    towards the subset of words that usually appear in method names.
    """

    def __init__(self, word_counter: WordCounter,
                 token_type_counter: WordCounter,
                 node_type_counter: WordCounter,
                 word_counter_labels: WordCounter):
        super().__init__(word_counter, token_type_counter, node_type_counter)
        self.word_counter_labels = word_counter_labels

    def __call__(self, sample: CTStage1Sample):
        super().__call__(sample)
        label_tokens = method_name_to_tokens(sample.func_name)
        for label_token in label_tokens:
            if str(label_token).strip(" \t\r").lower() != '':
                self.word_counter_labels.update(label_token)


class VocabularyTransformer:
    """
    Used in stage 2 to transform all samples into vocabularized versions, i.e., replace tokens, token_types and
    node_types
    with there respective IDs in the vocabulary.
    """

    def __init__(self, word_vocab, token_type_vocab, node_type_vocab):
        self.word_vocab = word_vocab
        self.token_type_vocab = token_type_vocab
        self.node_type_vocab = node_type_vocab

    def __call__(self, sample: CTStage1Sample):
        for token in sample.tokens:
            assert all([isinstance(st, str) for st in
                        token.sub_tokens]), f"Some sub tokens ({token.sub_tokens}) do not have string values. Has " \
                                            f"this sample already been vocabularized?"
            for i, st in enumerate(token.sub_tokens):
                token.sub_tokens[i] = self.word_vocab[st]
                token.original_sub_tokens[i] = st
            token.token_type = self.token_type_vocab[token.token_type]

        for node in sample.ast.nodes.values():
            node.node_type = self.node_type_vocab[node.node_type]
        return sample


class CodeSummarizationVocabularyTransformer(VocabularyTransformer):
    """
    Vocabulary Transformer extension for the Code Summarization task.
    Adds a `encoded_func_name` field to the sample that contains the IDs of the method name.
    """

    def __init__(self, word_vocab, token_type_vocab, node_type_vocab, word_vocab_labels):
        super().__init__(word_vocab, token_type_vocab, node_type_vocab)
        self.word_vocab_labels = word_vocab_labels

    def __call__(self, sample: CTStage1Sample):
        sample = super().__call__(sample)
        label_tokens = method_name_to_tokens(sample.func_name)
        sample.encoded_func_name = [self.word_vocab_labels[t] for t in label_tokens]
        return sample


def batch_decode(vocabulary: Vocabulary, token_ids):
    import numpy as np
    shape_before = token_ids.shape
    decoded = []
    tensor_resh = token_ids.reshape([-1, token_ids.shape[-1]])
    for token in tensor_resh:
        decoded.append("_".join([str(vocabulary.reverse_lookup(subtoken.item())) for subtoken in token]))
    decoded = np.array(decoded)
    decoded = decoded.reshape(shape_before[:-1])
    decoded = np.array([" ".join(x) for x in decoded])
    return decoded
