"""
A collection of classes for calculation distance metrics on graphs.
"""

from abc import ABC

import networkx as nx
import numpy as np
import torch

from code_transformer.modeling.constants import BIN_PADDING
from code_transformer.preprocessing.graph.alg import next_sibling_shortest_paths, all_pairs_shortest_paths, \
    tree_shortest_paths
from code_transformer.preprocessing.graph.binning import calculate_bins, hist_by_area, EqualBinning

UNREACHABLE = 10000


class GraphDistanceMetric(ABC):

    def __init__(self, name):
        self.name = name

    def __call__(self, adjacency_matrix: torch.tensor) -> torch.tensor:
        pass

    def get_name(self):
        return self.name


# =============================================================================
# Distance metrics
# =============================================================================


class AbstractShortestPaths(GraphDistanceMetric, ABC):

    def __init__(self, name, forward=True, backward=True, negative_reverse_dists=True, threshold=None):
        """
        :param threshold: any shortest path above this threshold will be regarded as UNREACHABLE (i.e., infinite distance)
        """
        super(AbstractShortestPaths, self).__init__(name)
        assert forward or backward
        self.forward = forward
        self.backward = backward
        self.negative_reverse_dists = negative_reverse_dists
        self.threshold = threshold

    def calculate_pw_dists(self, adjacency_matrix, sp_length):
        pw_dists = torch.ones_like(adjacency_matrix, dtype=torch.long) * UNREACHABLE
        pw_dists = pw_dists.long()
        pw_dists[sp_length[:, 0], sp_length[:, 1]] = sp_length[:, 2]
        if self.forward and self.backward:
            pw_dists = torch.triu(pw_dists, diagonal=1)
            pw_dists = pw_dists - pw_dists.t()
            if not self.negative_reverse_dists:
                pw_dists *= -(pw_dists == -UNREACHABLE).long() \
                            + (pw_dists != -UNREACHABLE).long()
        elif self.backward:
            pw_dists = pw_dists.t()
        if self.threshold:
            pw_dists[pw_dists.abs() > self.threshold] = UNREACHABLE
        return pw_dists


class AncestorShortestPaths(AbstractShortestPaths):
    name = "ancestor_sp"

    def __init__(self, forward=True, backward=True, negative_reverse_dists=True, threshold=None):
        super(AncestorShortestPaths, self).__init__(self.name, forward, backward, negative_reverse_dists, threshold)

    def __call__(self, adjacency_matrix: torch.tensor) -> torch.tensor:
        G = nx.from_numpy_array(torch.triu(adjacency_matrix).numpy(),
                                create_using=nx.DiGraph)

        sp_length = all_pairs_shortest_paths(G=G)
        pw_dists = self.calculate_pw_dists(adjacency_matrix, sp_length)
        return pw_dists


class SiblingShortestPaths(AbstractShortestPaths):
    name = "sibling_sp"

    def __init__(self, forward=True, backward=True, negative_reverse_dists=True, threshold=None):
        super(SiblingShortestPaths, self).__init__(self.name, forward, backward, negative_reverse_dists, threshold)

    def __call__(self, adjacency_matrix: torch.tensor) -> torch.tensor:
        edges = torch.triu(adjacency_matrix).nonzero()
        sp_length = next_sibling_shortest_paths(edges)
        pw_dists = self.calculate_pw_dists(adjacency_matrix, sp_length)
        return pw_dists


class ShortestPaths(GraphDistanceMetric):
    name = "shortest_paths"

    def __init__(self, threshold=None):
        super(ShortestPaths, self).__init__(self.name)
        self.threshold = threshold

    def __call__(self, adjacency_matrix: torch.tensor) -> torch.tensor:
        shortest_paths = tree_shortest_paths(adjacency_matrix).long()

        if self.threshold:
            shortest_paths[shortest_paths.abs() > self.threshold] = UNREACHABLE
        return shortest_paths


class PersonalizedPageRank(GraphDistanceMetric):
    name = "ppr"

    def __init__(self, log=True, alpha=0.15, threshold=None):
        """
        :param log: whether to return the log of the ppr values
        :param alpha: teleport probability
        :param threshold: Any ppr score below this threshold will be regarded as 0
        """
        super(PersonalizedPageRank, self).__init__(self.name)
        self.alpha = alpha
        self.log = log
        self.threshold = threshold

    def __call__(self, adjacency_matrix: torch.tensor) -> torch.tensor:
        """
        :param adjacency_matrix: can be multiple adjacency matrices. The first dimension remains untouched
        :return:
        """
        A = adjacency_matrix
        if len(A.shape) == 2:
            A = A.unsqueeze(0)
        N = A.shape[1]
        degs_inv = A.sum(-1) ** -1  # inverted row sum
        degs_inv[A.sum(-1) == 0] = 0  # nodes with no outgoing edge
        D_inv = torch.diag_embed(degs_inv)
        A_rw = A.transpose(1, 2) @ D_inv
        ppr_mat = self.alpha * torch.inverse((torch.eye(N).to(A_rw) - (1 - self.alpha) * A_rw))
        ppr_mat = ppr_mat.squeeze(0)
        if self.log:
            ppr_mat = -(ppr_mat + 1e-9).log()

        if self.threshold:
            if self.log:
                ppr_mat[ppr_mat > -np.log(self.threshold)] = UNREACHABLE
            else:
                ppr_mat[ppr_mat < self.threshold] = 0

        return ppr_mat


# =============================================================================
# Distance Binning
# =============================================================================

class DistanceBinning:

    def __init__(self, n_bins, n_fixed=9, trans_func=EqualBinning()):
        self.n_bins = n_bins
        self.n_fixed = n_fixed
        self.trans_func = trans_func

    def __call__(self, distance_matrix: torch.tensor):
        # continuous distances (personalized page rank)
        if distance_matrix.dtype in [torch.float32, torch.float16, torch.float64]:
            dist_values = distance_matrix.reshape(-1).cpu().numpy()
            if UNREACHABLE in dist_values:
                # We fix UNREACHABLE to be the bin with index 0
                possible_distances = torch.tensor(
                    hist_by_area(dist_values[dist_values < UNREACHABLE], self.n_bins - 2), dtype=torch.float32)
                possible_distances = torch.cat(
                    [torch.tensor([UNREACHABLE], dtype=torch.float32), possible_distances])
                dist_values = torch.tensor(dist_values, dtype=torch.float32)
                indices = (dist_values[:, None] > possible_distances[None, :]).sum(-1).reshape(distance_matrix.shape)
                # Shift all indices by 1, as UNREACHABLE will be bin 0
                indices = (indices + 1) % self.n_bins
            else:
                # We assume there is already many 0 values in the distance matrix, that will be in bin 0
                possible_distances = torch.tensor(hist_by_area(dist_values, self.n_bins - 1),
                                                  dtype=torch.float32)
                dist_values = torch.tensor(dist_values, dtype=torch.float32)
                # It is very important that both dist_values and possible_distances are float32
                # Otherwise, it can happen that a > b if a = b, just because a.dtype = float64, b.dtype=32
                indices = (dist_values[:, None] > possible_distances[None, :]).sum(-1).reshape(distance_matrix.shape)

        # discrete distances (shortest paths)
        elif distance_matrix.dtype in [torch.long, torch.int32, torch.int16, torch.int8, torch.int, torch.bool]:
            max_distance = 1000
            dist_values = distance_matrix.clamp(-max_distance, max_distance)
            unreachable_ixs = abs(dist_values) == max_distance
            dist_values[unreachable_ixs] = UNREACHABLE
            unique_values = dist_values.unique()
            num_unique_values = len(unique_values)
            num_unique_values += 1 if UNREACHABLE not in dist_values else 0
            num_unique_values = min(num_unique_values, self.n_bins)
            dist_values = dist_values.reshape(-1).cpu().numpy()

            values = dist_values[abs(dist_values) < max_distance]
            if num_unique_values < self.n_bins:
                # if there are fewer unique values than bins, we do not need to use sophisticated binning
                # possible_distances = hist_by_area(values, num_unique_values - 1)
                possible_distances = torch.sort(unique_values)[0]
                if UNREACHABLE in possible_distances:
                    possible_distances = possible_distances[:-1]
            else:
                if isinstance(self.trans_func, EqualBinning) and self.n_fixed == 0:
                    # Also resort to regular binning
                    possible_distances = hist_by_area(values, num_unique_values - 1)
                else:
                    # Calculate bins where the area of each bin is governed by trans_func.
                    # When using exponential binning, this means that bin area will grow exponentially in distance
                    # to the bin containing the value 0
                    possible_distances = calculate_bins(values, num_unique_values - 1, self.n_fixed, hist_by_area,
                                                        self.trans_func)

            if isinstance(possible_distances, torch.Tensor):
                possible_distances = possible_distances.type(torch.float32)
            else:
                possible_distances = torch.tensor(possible_distances, dtype=torch.float32)
            dist_values = torch.tensor(dist_values)

            neg_bins, pos_bins = possible_distances[possible_distances < 0], possible_distances[possible_distances >= 0]
            neg_vals, pos_vals = dist_values[dist_values < 0], dist_values[dist_values >= 0]
            neg_ixs = (neg_vals[:, None] >= neg_bins[None]).sum(-1) - 1
            pos_ixs = len(pos_bins) - (pos_vals[:, None] <= pos_bins[None]).sum(-1)

            indices = torch.cat([neg_ixs, pos_ixs + len(neg_bins)])
            indices = torch.zeros_like(dist_values)
            indices[dist_values < 0] = neg_ixs
            indices[dist_values >= 0] = pos_ixs + len(neg_bins)
            # Shift all indices by 1, as UNREACHABLE will be bin 0
            indices += 1
            indices = indices.reshape(distance_matrix.shape)
            indices[unreachable_ixs] = 0
            possible_distances = torch.cat([torch.tensor([UNREACHABLE], dtype=torch.float32), possible_distances])

            if num_unique_values < self.n_bins:
                bin_padding_tensor = torch.tensor([BIN_PADDING for i in range(self.n_bins - num_unique_values)],
                                                  dtype=torch.float32)
                possible_distances = torch.cat([possible_distances, bin_padding_tensor])

        else:
            raise NotImplementedError(f"Binning for tensors of type {distance_matrix.dtype} is not impolemented")

        assert 0 <= indices.min() and indices.max() <= self.n_bins, f"Indices have to be in [0, {self.n_bins}] but got [{indices.min()}, {indices.max()}]"
        assert len(
            possible_distances) == self.n_bins, f"Calculated amount of bins ({len(possible_distances)}) differs from requested amount ({self.n_bins})"

        return indices, possible_distances
