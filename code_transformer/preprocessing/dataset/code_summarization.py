from collections import namedtuple
from typing import List

import torch

from code_transformer.modeling.constants import CLS_TOKENS, CLS_TOKEN, PAD_TOKEN, MAX_NUM_TOKENS, UNKNOWN_TOKEN
from code_transformer.preprocessing.pipeline.filter import pad_or_truncate
from code_transformer.preprocessing.datamanager.base import CTBatch
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.base import CTBaseDataset
from code_transformer.preprocessing.nlp.text import RangeInterval
from code_transformer.preprocessing.nlp.tokenization import CTToken, split_identifier_into_parts, \
    get_idx_no_punctuation
from code_transformer.utils.log import get_logger
from code_transformer.utils.vocab import decode_tokens

logger = get_logger(__file__)

CTCodeSummarizationSample = namedtuple("CTCodeSummarizationSample", ["tokens", "token_types",
                                                                     "node_types", "distance_matrices",
                                                                     "binning_vectors", "distance_names",
                                                                     "func_name", "idx_func_tokens",
                                                                     "label", "extended_vocabulary",
                                                                     "extended_vocabulary_ids",
                                                                     "pointer_pad_mask", "language"])


class CTCodeSummarizationDataset(CTBaseDataset):
    """
    Processes the data for a code summarization task (which essentially is predicting method names).
    The main idea is to interpret the method name as a single token that should be predicted.
    For models without a decoder, a special [CLS] token is introduced that is supposed to capture the general context
    of a function. This is necessary, to be compatible with TransformerLanguageModel's interface.
    As it can happen that the function name does not appear in the data (because it is empty => anonymous functions)
    an additional token is inevitable.
    To prevent information leak, an attention mask is calculated where no position can attend the [CLS] token or the
    tokens corresponding to the function name. Only the function name tokens and the [CLS] token can attend the [CLS]
    token.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_sub_tokens_output=5, use_token_types=True,
                 use_pointer_network=False, max_num_tokens=MAX_NUM_TOKENS):
        super(CTCodeSummarizationDataset, self).__init__(data_manager, token_distances, max_distance_mask,
                                                         num_sub_tokens, use_token_types,
                                                         use_pointer_network=use_pointer_network,
                                                         max_num_tokens=max_num_tokens)
        self.num_sub_tokens_output = num_sub_tokens_output

    def __len__(self):
        return self.data_manager.approximate_total_samples()

    def __next__(self):
        sample = self._get_next_sample()

        cls_tokens = [self.word_vocab[CLS_TOKENS[i]] for i in range(self.num_sub_tokens)]
        cls_token = CTToken(cls_tokens, None, self.token_type_vocab[CLS_TOKEN])

        # It can happen that the sequence will become larger than MAX_NUM_TOKENS, which is okay. Worst case, is a
        # GPU out of memory
        sample.tokens.insert(0, cls_token)
        sample.token_mapping.insert(0, 0)  # Map inserted CLS Token to root node

        # Split function name into sub tokens
        func_name = sample.func_name
        func_name = func_name[func_name.rindex('.') + 1:] if '.' in func_name else func_name
        label = split_identifier_into_parts(func_name)
        if func_name == '':
            # Special case in JavaScript: anonymous functions with empty name. This can lead to data leakage, as the
            # AST will lack a node representing the method name. Hence, these samples have to be ignored for now

            # Ensure to free memory
            del sample, cls_tokens, cls_token, func_name, label
            return next(self)

        idx_func_tokens = []
        # Encode with ID and pad
        encoded_label = [self.word_vocab[sub_token] for sub_token in label]
        encoded_label = pad_or_truncate(encoded_label, self.num_sub_tokens_output, self.word_vocab[PAD_TOKEN])

        # Find the ascending indices of the tokens that correspond to the function name in order to mask them later
        # This is a bit complicated as function names can span several tokens but the tokens are already padded
        for i in range(len(sample.tokens)):
            at_least_one_match = False  # One sub token has to match in every subsequent token
            v_i = i  # "virtual" token index that simulates that a function name can span several tokens
            v_j = 0  # "virtual" label index that ensures that when this happens the next token has to match

            # the remaining label tokens by starting from the first sub token position
            # Example func_name: fooMethod=
            #         tokens: [1, 2, 0, 0, 0], [3, 0, 0, 0, 0]  (foo, method, PAD, PAD, PAD) (=, PAD, PAD, PAD, PAD)
            #         label: [1, 2, 3, 0, 0]                    (foo, method, =, PAD, PAD)
            label_idx = 0
            idx_func_tokens.append(i)

            while label_idx < min(self.num_sub_tokens, len(encoded_label)):
                if v_j < len(sample.tokens[v_i].sub_tokens) and sample.tokens[v_i].sub_tokens[v_j] == encoded_label[
                    label_idx]:
                    at_least_one_match = True
                    v_j += 1
                    label_idx += 1
                elif encoded_label[v_j] == self.word_vocab[PAD_TOKEN]:
                    # Label was completely matched, can exit
                    break
                elif at_least_one_match and v_j >= len(sample.tokens[v_i].sub_tokens):
                    # Reached end of first matching token
                    v_i += 1  # Check next token
                    v_j = 0  # For next token use again first sub token
                    at_least_one_match = False  # Ensure that next token has to have at least one match
                    if v_i >= len(sample.tokens):
                        # reached the end of the token sequence. Abort
                        idx_func_tokens = []
                        break
                    else:
                        idx_func_tokens.append(v_i)
                else:
                    idx_func_tokens = []
                    at_least_one_match = False
                    break  # Cannot match function name at this position
            if at_least_one_match:
                # Function name was matched, exit the loop
                break

        if not idx_func_tokens:
            logger.warn(f"Could not find method name {func_name} in token sequence, skipping sample")
            # logger.warn(f"Token sequence was: {str([str([self.word_vocab.reverse_lookup(st) for st in token.sub_tokens]) for token in sample.tokens])}")

            # Ensure to free memory
            del sample, cls_tokens, cls_token, func_name, label, idx_func_tokens, encoded_label
            return next(self)

        # Remove any additional tokens if label is more than 1 token long
        sample.tokens = [t for i, t in enumerate(sample.tokens) if i not in idx_func_tokens[1:]]
        sample.token_mapping = [m for i, m in enumerate(sample.token_mapping) if i not in idx_func_tokens[1:]]
        sample.tokens[idx_func_tokens[0]] = CTToken(
            [self.word_vocab[CLS_TOKENS[-1]]],
            RangeInterval.empty_interval(),
            self.token_type_vocab[UNKNOWN_TOKEN])

        # Replace label token with method name mask
        idx_func_tokens = [idx_func_tokens[0]]  # only one label position left now

        transformed_sample = self.transform_sample(sample)

        assert len(label) > 0, \
            f"Function name has to be comprised of at least 1 sub token, got {transformed_sample.func_name}"

        if hasattr(self, 'word_vocab_labels'):
            # We have a separate vocabulary for method name tokens
            encoded_label = pad_or_truncate(sample.encoded_func_name, self.num_sub_tokens_output,
                                            self.word_vocab[PAD_TOKEN])

        if self.use_pointer_network:
            # Encode label with extended vocabulary, essentially getting rid of unknown tokens in the label
            if hasattr(self, 'word_vocab_labels'):
                encoded_label = [(transformed_sample.extended_vocabulary[sub_token]
                                  if sub_token in transformed_sample.extended_vocabulary else
                                  self.unk_id)
                                 if sub_token not in self.word_vocab_labels
                                 else self.word_vocab_labels[sub_token]
                                 for sub_token in label]
            else:
                encoded_label = [(transformed_sample.extended_vocabulary[sub_token]
                                  if sub_token in transformed_sample.extended_vocabulary else
                                  self.unk_id)
                                 if sub_token not in self.word_vocab
                                 else self.word_vocab[sub_token]
                                 for sub_token in label]
            encoded_label = pad_or_truncate(encoded_label, self.num_sub_tokens_output, self.word_vocab[PAD_TOKEN])

            # Mask Method name tokens such that they cannot be pointed to
            n_label_subtokens = (transformed_sample.pointer_pad_mask[idx_func_tokens] == True).sum().item()
            assert n_label_subtokens == 1, f"Label should be only a masked label token, but was {transformed_sample.tokens[idx_func_tokens]}"
            transformed_sample.pointer_pad_mask[idx_func_tokens] = False

            # Remove method name tokens from extended vocabulary IDs such that they cannot be pointed to
            idx_func_sub_tokens = transformed_sample.pointer_pad_mask[:idx_func_tokens[0]].sum().item()
            del transformed_sample.extended_vocabulary_ids[idx_func_sub_tokens]

            # If a sub token only appears in the label, but nowhere else in the code, the model has no change to
            # predict it. Hence, it is truly unknown
            if hasattr(self, 'word_vocab_labels'):
                encoded_label_unk = [
                    sub_token if sub_token in transformed_sample.extended_vocabulary_ids or sub_token < len(
                        self.word_vocab_labels) else self.unk_id
                    for sub_token in encoded_label]
            else:
                encoded_label_unk = [
                    sub_token if sub_token in transformed_sample.extended_vocabulary_ids or sub_token < len(
                        self.word_vocab) else self.unk_id
                    for sub_token in encoded_label]
            assert encoded_label_unk == encoded_label, f"previously generated encoded label should note have contained any tokens that are only contained in the label, but got {encoded_label} vs {encoded_label_unk}"

            assert transformed_sample.pointer_pad_mask.sum() == len(transformed_sample.extended_vocabulary_ids), \
                f"Number of non-masked pointer sub tokens ({transformed_sample.pointer_pad_mask.sum()}) differs " \
                f"from number of subtokens in extended vocabulary IDs " \
                f"({len(transformed_sample.extended_vocabulary_ids)}). Extended vocabulary: {transformed_sample.extended_vocabulary_ids}, tokens: {transformed_sample.tokens}, pointer_pad_mask: {transformed_sample.pointer_pad_mask}, label: {encoded_label}"

        return CTCodeSummarizationSample(tokens=transformed_sample.tokens,
                                         token_types=transformed_sample.token_types,
                                         node_types=transformed_sample.node_types,
                                         distance_matrices=transformed_sample.distance_matrices,
                                         binning_vectors=transformed_sample.binning_vectors,
                                         distance_names=transformed_sample.distance_names,
                                         func_name=transformed_sample.func_name,
                                         idx_func_tokens=idx_func_tokens, label=encoded_label,
                                         extended_vocabulary=transformed_sample.extended_vocabulary,
                                         extended_vocabulary_ids=transformed_sample.extended_vocabulary_ids,
                                         pointer_pad_mask=transformed_sample.pointer_pad_mask,
                                         language=transformed_sample.language)

    def collate_fn(self, batch: List[CTCodeSummarizationSample]):
        collated_batch = super(CTCodeSummarizationDataset, self).collate_fn(batch)
        seq_len = collated_batch.tokens.shape[1]

        labels = []
        attention_masks = []
        target_mapping_per_token = []
        target_mapping = []

        for sample in batch:
            attention_mask = torch.zeros((seq_len, seq_len))
            attention_mask[:, 0] = 1  # No position can attend [CLS] token
            attention_mask[0, 0] = 0  # [CLS] can attend itself

            for idx in sample.idx_func_tokens:
                attention_mask[:, idx] = 1  # No position can attend func tokens
                attention_mask[idx, 0] = 0  # func tokens can attend [CLS]
            attention_masks.append(attention_mask)

            label = torch.tensor([sample.label])  # There is only one label per sample, the function name
            labels.append(label)

            t_mapping_per_token = torch.zeros(seq_len)
            t_mapping_per_token[0] = 1  # The [CLS] tokens at position 0 will be used for prediction
            target_mapping_per_token.append(t_mapping_per_token)

            t_mapping = torch.zeros(seq_len)
            t_mapping[0] = 1
            t_mapping = t_mapping.unsqueeze(0)
            target_mapping.append(t_mapping)

        return CTBatch(tokens=collated_batch.tokens, token_types=collated_batch.token_types,
                       node_types=collated_batch.node_types,
                       relative_distances=collated_batch.relative_distances,
                       distance_names=collated_batch.distance_names,
                       sequence_lengths=collated_batch.sequence_lengths, pad_mask=collated_batch.pad_mask,
                       labels=torch.stack(labels), perm_mask=torch.stack(attention_masks),
                       target_mapping=torch.stack(target_mapping),
                       target_mapping_per_token=torch.stack(target_mapping_per_token),
                       extended_vocabulary=collated_batch.extended_vocabulary,
                       extended_vocabulary_ids=collated_batch.extended_vocabulary_ids,
                       pointer_pad_mask=collated_batch.pointer_pad_mask,
                       languages=collated_batch.languages)


GreatBatch = namedtuple("GreatBatch", ['tokens', 'sequence_lengths', 'pad_mask', 'labels',
                                       'target_mapping', 'target_mapping_per_token', 'edge_ixs', 'attention_mask',
                                       "extended_vocabulary",
                                       "extended_vocabulary_ids",
                                       "pointer_pad_mask", 'languages'])


class CTCodeSummarizationDatasetEdgeTypes(CTCodeSummarizationDataset):
    """
    Processes the data for code summarization with GREAT.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_sub_tokens_output=5, edge_distance_names=None, use_pointer_network=False,
                 max_num_tokens=MAX_NUM_TOKENS):
        if edge_distance_names is None:
            edge_distance_names = ["shortest_paths", "ancestor_sp", "sibling_sp"]

        self.edge_distance_names = edge_distance_names
        super(CTCodeSummarizationDatasetEdgeTypes, self).__init__(data_manager, token_distances, max_distance_mask,
                                                                  num_sub_tokens, num_sub_tokens_output,
                                                                  use_pointer_network=use_pointer_network,
                                                                  max_num_tokens=max_num_tokens)

    def __next__(self):
        return super(CTCodeSummarizationDatasetEdgeTypes, self).__next__()

    def collate_fn(self, batch: List[CTCodeSummarizationSample]):
        collated_batch = super(CTCodeSummarizationDatasetEdgeTypes, self).collate_fn(batch)
        seq_len = collated_batch.tokens.shape[1]

        relative_distances = collated_batch.relative_distances
        edge_ixs_with_types = []
        num_edge_types = 0
        for ix, distance_name in enumerate(collated_batch.distance_names):
            if distance_name not in self.edge_distance_names:
                continue
            ixs, values = relative_distances[ix]
            if distance_name in ["ancestor_sp", "sibling_sp"]:
                vls_1 = (values.t() == -1).nonzero()
                nnz = (ixs == vls_1[:, 1].unsqueeze(-1).unsqueeze(-1)).nonzero().t()
                edge_ixs = torch.cat([torch.ones([1, nnz.shape[1]], dtype=torch.long) * num_edge_types, nnz], dim=0)
                edge_ixs_with_types.append(edge_ixs)
                num_edge_types += 1
                vls_2 = (values.t() == 1).nonzero()
                nnz_2 = (ixs == vls_2[:, 1].unsqueeze(-1).unsqueeze(-1)).nonzero().t()
                edge_ixs = torch.cat([torch.ones([1, nnz_2.shape[1]], dtype=torch.long) * num_edge_types, nnz_2], dim=0)
                edge_ixs_with_types.append(edge_ixs)
                num_edge_types += 1
            else:
                edge_ixs = (ixs == 1).nonzero().t()
                edge_ixs = torch.cat([torch.ones([1, edge_ixs.shape[1]], dtype=torch.long) * num_edge_types, edge_ixs],
                                     dim=0)
                edge_ixs_with_types.append(edge_ixs)
                num_edge_types += 1
        edge_ixs_with_types = torch.cat(edge_ixs_with_types, dim=1)

        return GreatBatch(tokens=collated_batch.tokens,
                          sequence_lengths=collated_batch.sequence_lengths,
                          pad_mask=collated_batch.pad_mask,
                          labels=collated_batch.labels,
                          target_mapping=collated_batch.target_mapping,
                          target_mapping_per_token=collated_batch.target_mapping_per_token,
                          edge_ixs=edge_ixs_with_types,
                          attention_mask=collated_batch.perm_mask,
                          extended_vocabulary=collated_batch.extended_vocabulary,
                          extended_vocabulary_ids=collated_batch.extended_vocabulary_ids,
                          pointer_pad_mask=collated_batch.pointer_pad_mask,
                          languages=collated_batch.languages)


class CTCodeSummarizationDatasetNoPunctuation(CTCodeSummarizationDataset):
    """
    Filters each sample to remove punctuation tokens like .,(): etc. as well as [INDENT]/[DEDENT] tokens.
    The idea is that for the code summarization task, these tokens are hardly important but instead elongate the token
    sequence unnecessarily.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_sub_tokens_output=5, use_token_types=True,
                 use_pointer_network=False, max_num_tokens=MAX_NUM_TOKENS):
        super(CTCodeSummarizationDatasetNoPunctuation, self).__init__(data_manager, token_distances, max_distance_mask,
                                                                      num_sub_tokens, num_sub_tokens_output,
                                                                      use_token_types,
                                                                      use_pointer_network=use_pointer_network,
                                                                      max_num_tokens=None)
        self.config = data_manager.load_config()
        self.max_num_tokens_no_punctuation = max_num_tokens

    def __next__(self):
        sample = super(CTCodeSummarizationDatasetNoPunctuation, self).__next__()

        # Calculate indices of tokens that should be kept, i.e., are tokens like identifiers or types
        decoded_tokens = decode_tokens(sample.tokens, word_vocab=self.word_vocab, config=self.config)
        idx = get_idx_no_punctuation(decoded_tokens)

        if self.max_num_tokens_no_punctuation is not None and len(idx) > self.max_num_tokens_no_punctuation + 1:
            # too many tokens
            logger.warn(f"Snippet has {len(idx)} tokens exceeding the limit of {self.max_num_tokens_no_punctuation}")
            return self.__next__()

        # In case there was a punctuation token before the function definition, the label ids have to be shifted as well
        idx_func_tokens_no_punctuation = []
        for idx_func_token in sample.idx_func_tokens:
            idx_func_tokens_no_punctuation.append(len([i for i in idx if i < idx_func_token]))

        # For the distance matrices, token sequence, token and node types, only indices corresponding to non punctuation
        # tokens are kept
        distance_matrices_no_punctuation = []
        for dist_matrix in sample.distance_matrices:
            distance_matrices_no_punctuation.append(dist_matrix[idx][:, idx])
        node_types_no_punctuation = sample.node_types[idx]
        token_types_no_punctuation = sample.token_types[idx]
        tokens_no_punctuation = sample.tokens[idx]

        pointer_pad_mask = sample.pointer_pad_mask
        extended_vocabulary_ids = sample.extended_vocabulary_ids
        if self.use_pointer_network:
            # Also remove punctuation tokens from extended_vocabulary_ids and pointer_pad_mask
            idx_sub_tokens = []
            current_sub_token = 0
            for i, mask in enumerate(sample.pointer_pad_mask):
                n_sub_tokens = mask.sum()
                if i in idx:
                    idx_sub_tokens.extend(range(current_sub_token, current_sub_token + n_sub_tokens))

                current_sub_token += n_sub_tokens

            pointer_pad_mask = sample.pointer_pad_mask[idx]
            extended_vocabulary_ids = [sample.extended_vocabulary_ids[i] for i in idx_sub_tokens]

            assert pointer_pad_mask.sum() == len(extended_vocabulary_ids), \
                f"Number of non-masked subtokens ({pointer_pad_mask.sum().item()}) does not match number of extended vocabulary ids ({len(extended_vocabulary_ids)})"

        return CTCodeSummarizationSample(tokens_no_punctuation, token_types_no_punctuation,
                                         node_types_no_punctuation, distance_matrices_no_punctuation,
                                         sample.binning_vectors, sample.distance_names, sample.func_name,
                                         idx_func_tokens_no_punctuation, sample.label,
                                         extended_vocabulary=sample.extended_vocabulary,
                                         extended_vocabulary_ids=extended_vocabulary_ids,
                                         pointer_pad_mask=pointer_pad_mask, language=sample.language)
