import torch

from code_transformer.modeling.constants import MAX_NUM_TOKENS, PAD_TOKEN
from code_transformer.preprocessing.pipeline.filter import pad_or_truncate
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.code_summarization import CTCodeSummarizationDataset
from code_transformer.preprocessing.nlp.tokenization import CTToken, split_identifier_into_parts


class CTCodeSummarizationOnlyASTDataset(CTCodeSummarizationDataset):
    """
    This ablation dataset ensures that the token sequence only contains information if the corresponding AST node
    is a leaf. In detail, a new token sequence is generated where the first token matches the first node, the second
    token the second node and so on.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_sub_tokens_output=5, use_pointer_network=False, max_num_tokens=MAX_NUM_TOKENS,
                 mask_all_tokens=False):
        super(CTCodeSummarizationOnlyASTDataset, self).__init__(data_manager, token_distances, max_distance_mask,
                                                                num_sub_tokens, num_sub_tokens_output,
                                                                use_token_types=False,
                                                                use_pointer_network=use_pointer_network,
                                                                max_num_tokens=None)
        self.max_num_tokens_only_ast = max_num_tokens
        self.config = data_manager.load_config()
        self.mask_all_tokens = mask_all_tokens

    def _inverse_token_mapping(self, token_mapping, i):
        return [k for k, v in enumerate(token_mapping) if v == i]

    def _decode_token(self, token):
        return [self.word_vocab.reverse_lookup(id) for id in token.string]

    def _filter_inverted_ids(self, sample, inverted_token_ids):
        inverted_token_ids = [id for id in inverted_token_ids if
                              sample.tokens[id].string[0] > len(self.config['preprocessing']['special_symbols'])]
        inverted_token_ids = [id for id in inverted_token_ids if len(sample.tokens[id].string) == max(
            [len(sample.tokens[id2].string) for id2 in inverted_token_ids])]
        inverted_token_ids = [id for id in inverted_token_ids if
                              len(self._decode_token(sample.tokens[id])[0]) == max(
                                  [len(self._decode_token(sample.tokens[id2])[0]) for id2 in
                                   inverted_token_ids])]
        return inverted_token_ids

    def _get_parent_token(self, sample, id):
        row = sample.graph_sample['adj'][id]
        parents = torch.where(row[:id] == 1)[0]
        if len(parents) == 0:
            return None
        parent_id = parents.item()
        parent_inverted_ids = self._inverse_token_mapping(sample.token_mapping, parent_id)
        parent_inverted_ids = self._filter_inverted_ids(sample, parent_inverted_ids)
        if len(parent_inverted_ids) == 0:
            return self._get_parent_token(sample, parent_id)
        token_text = "".join(self._decode_token(sample.tokens[parent_inverted_ids[0]]))
        alpha_chars = [c for c in token_text if c.isalpha()]
        if len(alpha_chars) / len(token_text) >= 0.5:
            return sample.tokens[parent_inverted_ids[0]]
        else:
            return self._get_parent_token(sample, parent_id)

    @staticmethod
    def _dummy_token():
        return CTToken([], None, 0)

    def _get_next_sample(self):
        sample = super(CTCodeSummarizationOnlyASTDataset, self)._get_next_sample()

        if self.max_num_tokens_only_ast is not None and sample.graph_sample['adj'].shape[
            0] > self.max_num_tokens_only_ast:
            return self._get_next_sample()

        new_sequence = []

        for i, row in enumerate(sample.graph_sample['adj']):
            n_children = row[(i + 1):].sum().item()
            if n_children == 0 and not self.mask_all_tokens:
                # Node is a leaf. Now, try to find the respective token
                inverted_token_ids = self._inverse_token_mapping(sample.token_mapping, i)
                if len(inverted_token_ids) == 0:
                    # No token was mapped to this leaf node.
                    new_sequence.append(self._dummy_token())
                elif len(inverted_token_ids) == 1:
                    # Desired case. There is exactly one token matched to this leaf node.
                    if inverted_token_ids[0] >= len(sample.tokens):
                        print(inverted_token_ids)
                        print(len(sample.tokens))
                        print(sample.token_mapping)
                    token = sample.tokens[inverted_token_ids[0]]
                    token.token_type = 0
                    new_sequence.append(token)
                else:

                    inverted_token_ids = self._filter_inverted_ids(sample, inverted_token_ids)
                    if len(inverted_token_ids) >= 1:
                        # There are multiple tokens matched to this leaf node. Just take the first then.
                        token = sample.tokens[inverted_token_ids[0]]
                        token.token_type = 0
                        new_sequence.append(token)
                    else:
                        new_sequence.append(self._dummy_token())
            else:
                # Node is not a leaf. Thus, the corresponding token will just be a DUMMY token
                new_sequence.append(CTToken([], None, 0))

        if self.mask_all_tokens:
            func_name = sample.func_name
            func_name = func_name[func_name.rindex('.') + 1:] if '.' in func_name else func_name
            label = split_identifier_into_parts(func_name)
            encoded_label = [self.word_vocab[sub_token] for sub_token in label]
            encoded_label = pad_or_truncate(encoded_label, self.num_sub_tokens, self.word_vocab[PAD_TOKEN])
            new_sequence[0].sub_tokens.extend(encoded_label)

        sample.tokens = new_sequence
        sample.token_mapping = list(range(len(new_sequence)))

        assert all([token.token_type == 0 for token in
                    sample.tokens]), f"AST ablation should not contain any token type information, but got {[token.token_type for token in sample.tokens]}"

        return sample
