from typing import List

import torch

from code_transformer.modeling.constants import SEP_TOKEN, CLS_TOKEN, MAX_NUM_TOKENS
from code_transformer.modeling.data_utils import sample_targets, permutation_attention_mask
from code_transformer.preprocessing.datamanager.base import CTBatch
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.dataset.base import CTBaseDataset, CTBaseSample
from code_transformer.preprocessing.nlp.tokenization import get_idx_no_punctuation
from code_transformer.utils.data import pad_list
from code_transformer.utils.vocab import decode_tokens


class CTLanguageModelingDataset(CTBaseDataset):
    """
    Dataset implementation to be used together with PyTorch's dataloader for general language modeling.
    Transforms the samples into torch tensors that fit into TransformerLanguageModel.
    Shuffling is not directly provided by the dataset but by the underlying data manager.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_labels_per_sample=5, max_num_tokens=MAX_NUM_TOKENS, use_pointer_network=False):
        """
        :param num_labels_per_sample: the number of tokens per sample to be predicted
        """

        super(CTLanguageModelingDataset, self).__init__(data_manager, token_distances, max_distance_mask,
                                                        num_sub_tokens, max_num_tokens=max_num_tokens,
                                                        use_pointer_network=use_pointer_network)
        self.num_labels_per_sample = num_labels_per_sample

    def collate_fn(self, batch: List) -> CTBatch:
        """
        Combines the given list of samples into a batch, taking care of correctly padding every tensor.
        Implements dynamic padding, i.e., sequences (and thus distance matrices) are padded with respect to the longest
        sequence in the batch and not a global padding value.
        """

        batch = super(CTLanguageModelingDataset, self).collate_fn(batch)

        seq_lengths = batch.sequence_lengths
        seq_tensors = batch.tokens
        max_distance_masks = batch.max_distance_mask
        padding_mask = batch.pad_mask
        max_seq_length = seq_tensors.shape[1]

        target_mapping, target_mapping_per_token = sample_targets(num_predict=self.num_labels_per_sample,
                                                                  seq_len=max_seq_length,
                                                                  batch_size=batch.tokens.shape[0],
                                                                  pad_mask=padding_mask)
        perm = permutation_attention_mask(seq_tensors, target_mapping_per_token,
                                          max_seq_length, max_seq_length, sep_id=self.word_vocab.vocabulary[SEP_TOKEN],
                                          cls_id=self.word_vocab.vocabulary[CLS_TOKEN])
        perm_mask = perm[0].long()
        perm_mask |= max_distance_masks  # Merge max distance attention mask with regular attention mask

        label_selected = seq_tensors.unsqueeze(-1) * target_mapping.transpose(1, 2).unsqueeze(2)
        labels = label_selected.max(1)[0].transpose(1, 2).long().contiguous()

        extended_vocabulary_ids = batch.extended_vocabulary_ids
        if self.use_pointer_network:
            extended_vocabulary_ids = []
            for idx_sample in range(batch.tokens.shape[0]):
                idx_func_tokens = torch.where(target_mapping_per_token[idx_sample] == 1)[0]
                current_pos = 0
                idx_func_sub_tokens = []
                for j, mask in enumerate(batch.pointer_pad_mask[idx_sample]):
                    n_sub_tokens = mask.sum().item()
                    if j in idx_func_tokens:
                        idx_func_sub_tokens.extend(range(current_pos, current_pos + n_sub_tokens))
                    current_pos += n_sub_tokens
                extended_vocabulary_ids.append([v_id.item() for j, v_id in enumerate(batch.extended_vocabulary_ids[idx_sample]) if j not in idx_func_sub_tokens and v_id.item() != self.sequence_pad_value])
                batch.pointer_pad_mask[idx_sample][idx_func_tokens] = False
                assert len(extended_vocabulary_ids[-1]) == batch.pointer_pad_mask[idx_sample].sum().item(), "number of sub tokens in extended_vocabulary_ids does not match number of non-masked pointer sub tokens"
            seq_len_subtokens = max([len(evi) for evi in extended_vocabulary_ids])
            extended_vocabulary_ids = torch.tensor([
                pad_list(evi, seq_len_subtokens, self.sequence_pad_value) for evi in extended_vocabulary_ids])

        return CTBatch(tokens=seq_tensors, token_types=batch.token_types, node_types=batch.node_types,
                       relative_distances=batch.relative_distances, distance_names=batch.distance_names,
                       sequence_lengths=seq_lengths, pad_mask=batch.pad_mask, labels=labels,
                       perm_mask=perm_mask, target_mapping=target_mapping,
                       target_mapping_per_token=target_mapping_per_token,
                       extended_vocabulary=batch.extended_vocabulary,
                       extended_vocabulary_ids=extended_vocabulary_ids,
                       pointer_pad_mask=batch.pointer_pad_mask, languages=batch.languages)


class CTLanguageModelingDatasetNoPunctuation(CTLanguageModelingDataset):
    """
    Filters each sample to remove punctuation tokens like .,(): etc. as well as [INDENT]/[DEDENT] tokens.
    The idea is that for the code summarization task, these tokens are hardly important but instead elongate the token
    sequence unnecessarily.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, num_labels_per_sample=5, min_sequence_length=5, max_num_tokens=MAX_NUM_TOKENS,
                 use_pointer_network=False):
        super(CTLanguageModelingDatasetNoPunctuation, self).__init__(data_manager, token_distances=token_distances,
                                                                     max_distance_mask=max_distance_mask,
                                                                     num_sub_tokens=num_sub_tokens,
                                                                     num_labels_per_sample=num_labels_per_sample,
                                                                     max_num_tokens=None,
                                                                     use_pointer_network=use_pointer_network)
        self.config = data_manager.load_config()
        self.min_sequence_length = min_sequence_length
        self.max_num_tokens_no_punctuation = max_num_tokens

    def __next__(self):
        sample = super(CTLanguageModelingDatasetNoPunctuation, self).__next__()

        # Calculate indices of tokens that should be kept, i.e., are tokens like identifiers or types
        decoded_tokens = decode_tokens(sample.tokens, word_vocab=self.word_vocab, config=self.config)
        idx = get_idx_no_punctuation(decoded_tokens)

        if len(idx) > self.max_num_tokens_no_punctuation + 1:
            return self.__next__()

        # For the distance matrices, token sequence, token and node types, only indices corresponding to non punctuation
        # tokens are kept
        distance_matrices_no_punctuation = []
        for dist_matrix in sample.distance_matrices:
            distance_matrices_no_punctuation.append(dist_matrix[idx][:, idx])
        node_types_no_punctuation = sample.node_types[idx]
        token_types_no_punctuation = sample.token_types[idx]
        tokens_no_punctuation = sample.tokens[idx]

        if len(tokens_no_punctuation) < self.num_labels_per_sample \
                or len(tokens_no_punctuation) < self.min_sequence_length:
            return next(self)

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

        return CTBaseSample(tokens_no_punctuation, token_types_no_punctuation, node_types_no_punctuation,
                            distance_matrices_no_punctuation, sample.binning_vectors, sample.distance_names,
                            sample.func_name, sample.docstring, sample.extended_vocabulary,
                            extended_vocabulary_ids, pointer_pad_mask, sample.language)
