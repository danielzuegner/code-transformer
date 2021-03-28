from collections import namedtuple
from typing import List, Union

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader

from code_transformer.modeling.constants import PAD_TOKEN, EOS_TOKEN, MAX_NUM_TOKENS, UNKNOWN_TOKEN
from code_transformer.modeling.data_utils import pad_mask
from code_transformer.preprocessing.pipeline.stage2 import CTStage2Sample, CTStage2MultiLanguageSample
from code_transformer.preprocessing.datamanager.preprocessed import CTPreprocessedDataManager
from code_transformer.preprocessing.nlp.tokenization import split_identifier_into_parts
from code_transformer.utils.data import pad_list

CTBaseSample = namedtuple("CTBaseSample", ["tokens", "token_types", "node_types", "distance_matrices",
                                           "binning_vectors", "distance_names", "func_name", "docstring",
                                           "extended_vocabulary", "extended_vocabulary_ids",
                                           "pointer_pad_mask", "language"])
CTBaseBatch = namedtuple("CTBaseBatch", ["tokens", "token_types", "node_types", "relative_distances",
                                         "distance_names", "sequence_lengths", "pad_mask",
                                         "max_distance_mask", "extended_vocabulary",
                                         "extended_vocabulary_ids", "pointer_pad_mask", "languages"])


class CTBaseDataset(IterableDataset):
    """
    Unites common functionalities used across different datasets such as applying the token mapping to the
    distance matrices and collating the matrices from multiple samples into one big tensor.
    """

    def __init__(self, data_manager: CTPreprocessedDataManager, token_distances=None, max_distance_mask=None,
                 num_sub_tokens=5, use_token_types=True, use_pointer_network=False,
                 max_num_tokens=MAX_NUM_TOKENS):
        self.data_manager = data_manager
        vocabularies = self.data_manager.load_vocabularies()
        if len(vocabularies) == 3:
            self.word_vocab, self.token_type_vocab, self.node_type_vocab = vocabularies
        else:
            self.word_vocab, self.token_type_vocab, self.node_type_vocab, self.word_vocab_labels = vocabularies
        self.token_distances = token_distances
        self.max_distance_mask = max_distance_mask
        self.sub_token_pad_value = self.word_vocab.vocabulary[PAD_TOKEN]
        self.sequence_pad_value = self.word_vocab.vocabulary[EOS_TOKEN]
        self.unk_id = self.word_vocab.vocabulary[UNKNOWN_TOKEN]
        self.token_type_pad_value = self.token_type_vocab.vocabulary[EOS_TOKEN]
        self.node_type_pad_value = self.node_type_vocab.vocabulary[EOS_TOKEN]
        self.num_sub_tokens = num_sub_tokens
        self.use_token_types = use_token_types
        self.use_pointer_network = use_pointer_network
        self.max_num_tokens = max_num_tokens

    def to_dataloader(self):
        return DataLoader(self, collate_fn=self.collate_fn)

    def __next__(self):
        sample = self._get_next_sample()
        return self.transform_sample(sample)

    def _get_next_sample(self):
        sample = next(self.dataset)

        if self.max_num_tokens is not None:
            if len(sample.tokens) > self.max_num_tokens:
                print(
                    f"Snippet has {len(sample.tokens)} tokens exceeding the limit of {self.max_num_tokens}")
                del sample
                return self._get_next_sample()

        sorted_mappings = sorted(sample.token_mapping.items(), key=lambda x: x[0])
        sample.token_mapping = [m[1] for m in sorted_mappings]

        return sample

    def transform_sample(self, sample: Union[CTStage2Sample, CTStage2MultiLanguageSample]):
        """
        Transforms a sample into torch tensors, applies sub token padding (which is independent of the other samples in
        a batch) and applies the token mapping onto the distance matrices (which makes them much bigger)
        """
        token_mapping = sample.token_mapping

        sequence = torch.tensor(
            [pad_list(token.sub_tokens, self.num_sub_tokens, self.sub_token_pad_value) for token in
             sample.tokens])
        if self.use_token_types:
            token_types = torch.tensor([token.token_type for token in sample.tokens])
        else:
            token_types = None

        node_types = torch.tensor(sample.graph_sample['node_types'])

        distance_matrices, binning_vectors, distance_names = zip(
            *[(distance[0], distance[1], distance[-1]) for distance in sample.graph_sample['distances']])

        binning_vectors = list(binning_vectors)  # Transform to list, to be able to append more bins
        distance_names = list(distance_names)

        # Apply mapping to graph objects (distance matrices and node type sequence)
        # This is as simple as duplicating certain rows and columns according to the token mapping
        # E.g., token mapping   + Distance matrix
        #       0 -> 1                                  | e e d f d |
        #       1 -> 1            | a b c |             | e e d f d |
        #       2 -> 0          + | d e f |         =>  | b b a c a |
        #       3 -> 2            | g h i |             | h h g i g |
        #       4 -> 0                                  | b b a c a |
        # This can be achieved by just using the token mapping values as index list for the distance tensor
        # mapped_nodes = list(token_mapping.values())
        mapped_nodes = token_mapping
        distance_matrices = [dist_matrix[mapped_nodes][:, mapped_nodes] for dist_matrix in distance_matrices]
        node_types = node_types[mapped_nodes]

        # Optionally, add the token distances metric
        if self.token_distances:
            indices, bins = self.token_distances(sequence, token_types, node_types)
            distance_matrices.append(indices)
            binning_vectors.append(bins)
            distance_names.append(self.token_distances.get_name())

        assert len({dist_matrix.shape[0] for dist_matrix in
                    distance_matrices}) == 1, "Distance matrices have differing lengths"

        extended_vocabulary = None
        extended_vocabulary_ids = None
        pointer_pad_mask = None
        if self.use_pointer_network:
            # Generating extended vocabulary for pointer network. The extended vocabulary essentially mimics an infinite
            # vocabulary size for this sample
            extended_vocabulary_ids = []
            extended_vocabulary = dict()
            len_vocab = len(self.word_vocab_labels) if hasattr(self, 'word_vocab_labels') else len(self.word_vocab)
            for idx_token, token in enumerate(sample.tokens):
                for idx_subtoken, subtoken in enumerate(token.sub_tokens):
                    if hasattr(self, 'word_vocab_labels'):
                        # Use output vocabulary for extended vocabulary in order to allow Pointer Network to merge
                        # regular predictions (output vocabulary) with pointers to input tokens (subtoken)
                        subtoken = self.word_vocab_labels[self.word_vocab.reverse_lookup(subtoken)]

                    if subtoken == self.unk_id:
                        if hasattr(token, 'original_sub_tokens'):
                            original_subtoken = token.original_sub_tokens[idx_subtoken]
                        else:
                            # Try reconstructing original sub token by using encoded sub token and vocabulary
                            original_subtoken = split_identifier_into_parts(
                                token.source_span.substring(sample.stripped_code_snippet))[idx_subtoken]

                        if original_subtoken in extended_vocabulary:
                            extended_id = extended_vocabulary[original_subtoken]
                        else:

                            extended_id = len_vocab + len(extended_vocabulary)
                            extended_vocabulary[original_subtoken] = extended_id
                        extended_vocabulary_ids.append(extended_id)
                    else:
                        extended_vocabulary_ids.append(subtoken)

            pointer_pad_mask = sequence != self.sub_token_pad_value

        sample_language = None
        if isinstance(sample, CTStage2MultiLanguageSample):
            # use multi-language
            sample_language = self.data_manager.language.split(',').index(sample.language)

        return CTBaseSample(tokens=sequence, token_types=token_types, node_types=node_types,
                            distance_matrices=distance_matrices, binning_vectors=binning_vectors,
                            distance_names=distance_names, func_name=sample.func_name, docstring=sample.docstring,
                            extended_vocabulary=extended_vocabulary,
                            extended_vocabulary_ids=extended_vocabulary_ids, pointer_pad_mask=pointer_pad_mask,
                            language=sample_language)

    def __iter__(self):
        # get a new generator from CTPreprocessedDataManager
        self.dataset = self.data_manager.read()
        return self

    def iter_from(self, samples):
        self.dataset = iter(samples)
        return self

    def collate_fn(self, batch: List[CTBaseSample]):
        max_seq_length = max([sample[0].shape[0] for sample in batch])  # length of largest sequence in batch
        # We use +1 here as in some occasions a [CLS] token is inserted at the beginning which increases sequence length
        assert self.max_num_tokens is None or max_seq_length <= self.max_num_tokens + 1, \
            f"Sample with more tokens than TOKENS_THRESHOLD ({self.max_num_tokens}): " \
            f"{[len(sample[0]) for sample in batch if sample[0].shape[0] > self.max_num_tokens]}"

        seq_tensors = []
        token_type_tensors = []
        node_type_tensors = []
        seq_lengths = []
        max_distance_masks = []
        relative_distances = dict()
        binning_vector_tensors = dict()
        for sample in batch:
            sequence = sample.tokens
            token_types = sample.token_types
            node_types = sample.node_types
            distance_matrices = sample.distance_matrices
            binning_vectors = sample.binning_vectors
            distance_names = sample.distance_names

            sequence_length = sequence.shape[0]
            seq_lengths.append(sequence_length)
            # indicates how much token, token type and node type sequences have to be padded
            pad_length = max_seq_length - sequence_length

            # Pad sequences
            sequence = F.pad(sequence, [0, 0, 0, pad_length], value=self.sequence_pad_value)
            if self.use_token_types:
                token_types = F.pad(token_types, [0, pad_length], value=self.token_type_pad_value)
            node_types = F.pad(node_types, [0, pad_length], value=self.node_type_pad_value)

            # Calculate and pad max distance mask
            max_distance_mask = torch.zeros_like(distance_matrices[0])
            if self.max_distance_mask:
                max_distance_mask = self.max_distance_mask(distance_matrices, binning_vectors, distance_names)
            max_distance_mask = F.pad(max_distance_mask, [0, pad_length, 0, pad_length], value=1)
            max_distance_masks.append(max_distance_mask)

            # Every sample has a matrix for every distance type. We want to have all matrices of the same distance
            # type grouped together in one dictionary entry (with the dictionary key being the indices of distance
            # types)
            for i, (dist_matrix, binning_vector) in enumerate(zip(distance_matrices, binning_vectors)):
                padded_dist_matrix = F.pad(dist_matrix,
                                           [0, pad_length, 0, pad_length])  # pad distance matrices with 0 bin
                if i not in relative_distances:
                    relative_distances[i] = []
                    binning_vector_tensors[i] = []
                relative_distances[i].append(padded_dist_matrix)
                binning_vector_tensors[i].append(binning_vector)

            # Group together sequences
            seq_tensors.append(sequence)
            if self.use_token_types:
                token_type_tensors.append(token_types)
            node_type_tensors.append(node_types)

        # Transform distance matrices and binning vectors into tensors
        for i in relative_distances.keys():
            relative_distances[i] = torch.stack(relative_distances[i])  # yields batch_size x N x N
            binning_vector_tensors[i] = torch.stack(binning_vector_tensors[i]).T  # yields K x batch_size

        seq_tensors = torch.stack(seq_tensors)
        seq_lengths = torch.tensor(seq_lengths)
        max_distance_masks = torch.stack(max_distance_masks)
        padding_mask = pad_mask(seq_lengths, max_len=max_seq_length)
        if self.use_token_types:
            token_type_tensors = torch.stack(token_type_tensors)
        else:
            token_type_tensors = None

        extended_vocabulary_ids = None
        extended_vocabulary = None
        pointer_pad_mask = None
        if self.use_pointer_network:
            # Simply pad extended_vocabulary IDs
            seq_len_subtokens = max([len(sample.extended_vocabulary_ids) for sample in batch])
            extended_vocabulary_ids = torch.tensor([
                pad_list(sample.extended_vocabulary_ids, seq_len_subtokens, self.sequence_pad_value)
                for sample in batch])

            pointer_pad_mask = pad_sequence([sample.pointer_pad_mask for sample in batch], batch_first=True,
                                            padding_value=False)

            extended_vocabulary = [v.extended_vocabulary for v in batch]

        languages = None
        if batch[0].language is not None:
            languages = torch.tensor([sample.language for sample in batch])

        return CTBaseBatch(tokens=seq_tensors, token_types=token_type_tensors,
                           node_types=torch.stack(
                               node_type_tensors), relative_distances=list(zip(relative_distances.values(),
                                                                               binning_vector_tensors.values())),
                           distance_names=distance_names, sequence_lengths=seq_lengths, pad_mask=padding_mask,
                           max_distance_mask=max_distance_masks, extended_vocabulary_ids=extended_vocabulary_ids,
                           pointer_pad_mask=pointer_pad_mask, extended_vocabulary=extended_vocabulary,
                           languages=languages)
