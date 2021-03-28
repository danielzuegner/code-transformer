from typing import List

import torch

from code_transformer.modeling.constants import PAD_TOKEN
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager


def decode_tokens(tokens: torch.Tensor, data_manager: CTPreprocessedDataManager = None, word_vocab=None,
                  config=None) -> List[List[str]]:
    assert data_manager is not None or word_vocab is not None and config is not None, "Either data_manager or word_vocab and config have to be provided"
    if word_vocab is None:
        word_vocab, _, _ = data_manager.load_vocabularies()
    if config is None:
        config = data_manager.load_config()
    pad_id = config['preprocessing']['special_symbols'][PAD_TOKEN]

    words = []
    for token in tokens:
        if isinstance(token, list) or isinstance(token, torch.Tensor):
            words.append(
                [word_vocab.reverse_lookup(sub_token.item()) for sub_token in token if not sub_token == pad_id])
        elif not token == pad_id:
            words.append(word_vocab.reverse_lookup(token))

    return words
