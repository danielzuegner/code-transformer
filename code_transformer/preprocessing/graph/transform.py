from typing import List, Dict

import networkx as nx
import torch

from code_transformer.preprocessing.pipeline.stage1 import CTStage1Sample
from code_transformer.preprocessing.pipeline.stage2 import CTStage2Sample
from code_transformer.preprocessing.graph.alg import preprocess_adj
from code_transformer.preprocessing.graph.distances import DistanceBinning, GraphDistanceMetric


class DistancesTransformer:
    """
    Applies the specified list of distance transformations to a sample
    """

    def __init__(self, distance_metrics: List[GraphDistanceMetric], distance_binning: DistanceBinning = None):
        self.distance_metrics = distance_metrics
        self.distance_binning = distance_binning

    def __call__(self, sample: CTStage1Sample) -> CTStage2Sample:
        G = sample.ast.to_networkx(create_using=nx.Graph)
        adj = torch.tensor(nx.to_numpy_matrix(G))
        graph_sample = {
            'adj': adj,
            'node_types': [node.node_type for node in sample.ast.nodes.values()],
            'distances': {}
        }
        for distance_metric in self.distance_metrics:
            distance_matrix = distance_metric(adj)
            if self.distance_binning:
                indices, bins = self.distance_binning(distance_matrix)
                distance_matrix = (indices, bins, distance_metric.get_name())
            graph_sample['distances'][distance_metric.get_name()] = distance_matrix

        if self.distance_binning:
            graph_sample['distances'] = list(graph_sample['distances'].values())

        return CTStage2Sample(sample.tokens, graph_sample, sample.token_mapping, sample.stripped_code_snippet,
                              sample.func_name, sample.docstring,
                              sample.encoded_func_name if hasattr(sample, 'encoded_func_name') else None)


# =============================================================================
# Experiment-Time distance transforms
# =============================================================================

class TokenDistancesTransform:
    name = "token_distances"

    def __init__(self, distance_binning: DistanceBinning):
        self.distance_binning = distance_binning

    def __call__(self, sequence, token_types, node_types):
        seq_len = len(sequence)
        rng = torch.arange(seq_len)
        diffs = rng[None, :] - rng[:, None]
        indices, bins = self.distance_binning(diffs)

        return indices, bins

    def get_name(self):
        return self.name


class MaxDistanceMaskTransform:
    """
    Calculates an attention mask for a sample by masking nodes that are further away than a specified maximum distance.
    Each distance type can have a different maximum distance. Use -1 to indicate that there is no maximum distance for
    this distance type.
    The generated binary attention mask is a matrix of size of the input distance matrices and uses 1 to indicate that
    a node exceeded max distance for ANY of the distance types (logical or)
    """

    def __init__(self, max_dist_per_type: Dict[str, float], agg_att_masks='or'):
        """
        :param max_dist_per_type: specifies the maximum distance for every distance type. Use -1 to indicate there is
             no max distance for that type. Every distance type has to be specified
        :param agg_att_masks: one of 'or', 'and'. How the attention masks from the individual distance types should
            be aggregated
        """
        self.max_dist_per_type = max_dist_per_type
        assert agg_att_masks in {'or', 'and'}, f"agg_att_masks can only be one of 'or', 'and'. Got {agg_att_masks}"
        self.agg_att_masks = agg_att_masks

    def __call__(self, dist_matrices: List[torch.Tensor], binning_vectors: List[torch.Tensor],
                 dist_names: List[str]) -> torch.Tensor:
        attention_mask = torch.zeros_like(dist_matrices[0])
        if self.agg_att_masks == 'and':
            attention_mask += 1
        for dist_matrix, binning_vector, dist_name in zip(dist_matrices, binning_vectors, dist_names):

            assert dist_name in self.max_dist_per_type, f"No max distance specified for {dist_name}"
            max_distance = self.max_dist_per_type[dist_name]
            if max_distance == -1:
                # Do not use this distance matrix for calculating the attention mask
                continue

            att_mask = torch.zeros_like(dist_matrix)
            idx_far_away = torch.where(binning_vector[dist_matrix].abs() > max_distance)
            att_mask[idx_far_away] = 1
            if self.agg_att_masks == 'or':
                attention_mask |= att_mask
            elif self.agg_att_masks == 'and':
                attention_mask &= att_mask

        return attention_mask


# =============================================================================
# Other transformations
# =============================================================================

class PreprocessAdj:
    """
    Normalizes all adjacency matrices
    """

    def __init__(self, replace_adj=True):
        self.replace_adj = replace_adj
        pass

    def __call__(self, sample, ):
        adjs = sample['adjs']
        adj_prep = preprocess_adj(adjs['adj'])
        if self.replace_adj:
            adjs['adj'] = adj_prep
        else:
            adjs['adj_prep'] = adj_prep
        return sample
