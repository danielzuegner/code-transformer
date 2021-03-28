import unittest

from code_transformer.modeling.constants import BIN_PADDING
from code_transformer.preprocessing.graph.distances import DistanceBinning, UNREACHABLE
import torch


class TestBinning(unittest.TestCase):

    def test_with_unreachable(self):
        values = torch.arange(-5, 6)
        values = torch.cat([values, torch.tensor([1000])])
        n_bins = 8
        DB = DistanceBinning(n_bins, n_fixed=3)
        ixs, bins = DB(values.to(torch.long))
        assert bins.allclose(torch.tensor([UNREACHABLE, -5, -3.5, -1, 0, 1, 3.5, 5]))
        assert bins[ixs].allclose(torch.tensor([-5, -5, -3.5, -3.5, -1, 0, 1, 3.5, 3.5, 5, 5, UNREACHABLE]))

    def test_without_unreachable(self):
        values = torch.arange(-5, 6)
        n_bins = 8
        DB = DistanceBinning(n_bins, n_fixed=3)
        ixs, bins = DB(values.to(torch.long))
        assert bins.allclose(torch.tensor([UNREACHABLE, -5, -3.5, -1, 0, 1, 3.5, 5]))
        assert bins[ixs].allclose(torch.tensor([-5, -5, -3.5, -3.5, -1, 0, 1, 3.5, 3.5, 5, 5]))

    def test_all_fixed(self):
        values = torch.arange(-5, 6)
        n_bins = len(values) + 1  # account for the UNREACHABLE bin
        DB = DistanceBinning(n_bins, n_fixed=len(values))
        ixs, bins = DB(values.to(torch.long))
        # bins should be:
        # [UNREACHABLE, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        assert bins.to(torch.long).allclose(torch.cat([torch.tensor([UNREACHABLE], dtype=torch.long), values]))
        # binned values should be:
        # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        assert bins[ixs].allclose(values.to(torch.float32))

    def test_all_fixed_even_number(self):
        values = torch.arange(-5, 5)
        n_bins = len(values) + 1  # account for the UNREACHABLE bin
        DB = DistanceBinning(n_bins, n_fixed=len(values))
        ixs, bins = DB(values.to(torch.long))
        # bins should be:
        # [UNREACHABLE, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4]
        assert bins.to(torch.long).allclose(torch.cat([torch.tensor([UNREACHABLE], dtype=torch.long), values]))
        # binned values should be:
        # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert bins[ixs].allclose(values.to(torch.float32))

    def test_all_fixed_with_unreachable(self):
        values_orig = torch.arange(-5, 6)
        values = torch.cat([values_orig, torch.tensor([1000]), torch.tensor([-1000])])
        n_bins = len(values) - 1
        DB = DistanceBinning(n_bins, n_fixed=len(values) - 2)
        ixs, bins = DB(values.to(torch.long))
        # bins should be:
        # [UNREACHABLE, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        assert bins.to(torch.long).allclose(torch.cat([torch.tensor([UNREACHABLE], dtype=torch.long), values_orig]))
        # binned values should be:
        # [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, UNREACHABLE, UNREACHABLE]
        assert bins[ixs].allclose(torch.cat([values_orig, torch.tensor([UNREACHABLE]),
                                             torch.tensor([UNREACHABLE])]).to(torch.float32))

    def test_all_fixed_with_unreachable_alternative(self):
        values_orig = torch.arange(-50, 51)
        values = torch.cat([values_orig, torch.tensor([1000])])
        n_bins = len(values)
        DB = DistanceBinning(n_bins, n_fixed=len(values) - 1)
        ixs, bins = DB(values.to(torch.long))
        assert bins.to(torch.long).allclose(torch.cat([torch.tensor([UNREACHABLE], dtype=torch.long), values_orig]))
        assert bins[ixs].allclose(torch.cat([values_orig, torch.tensor([UNREACHABLE])]).to(torch.float32))

    def test_mixed_positive_negative(self):
        values = torch.tensor([5, -4, 3, -2, 1, 0, 8, 8, -8, -8])
        n_bins = 6
        DB = DistanceBinning(n_bins, n_fixed=3)
        ixs, bins = DB(values.to(torch.long))

        self.assertTrue(bins.allclose(torch.tensor([UNREACHABLE, -8, -1, 0, 1, 8], dtype=torch.float)))
        self.assertTrue(bins[ixs].allclose(torch.tensor([8, -8, 8, -8, 1, 0, 8, 8, -8, -8], dtype=torch.float)))

    def test_2d_matrix(self):
        values = torch.arange(-6, 7, step=1).unsqueeze(0).repeat((7, 1))
        n_bins = 10
        DB = DistanceBinning(n_bins, n_fixed=5)
        ixs, bins = DB(values.to(torch.long))

        self.assertTrue(bins[ixs][0].allclose(
            torch.tensor([-6, -6, -4.5, -4.5, -2, -1, 0, 1, 2, 4.5, 4.5, 6, 6], dtype=torch.float)))

    def test_fewer_unique_values_than_bins(self):
        values = torch.arange(-10, 11, step=5).unsqueeze(0).repeat((3, 1))
        n_bins = 32
        DB = DistanceBinning(n_bins, n_fixed=5)
        ixs, bins = DB(values.to(torch.long))

        self.assertTrue(bins.allclose(torch.cat([torch.tensor([UNREACHABLE, -10, -5, 0, 5, 10], dtype=torch.float),
                                                 torch.tensor([BIN_PADDING], dtype=torch.float).expand(26)])))

    def test_uneven_bins(self):
        values = torch.arange(-10, 11, step=1)
        n_bins = 7
        DB = DistanceBinning(n_bins, n_fixed=5)
        ixs, bins = DB(values.to(torch.long))

        self.assertTrue(bins.allclose(torch.tensor([UNREACHABLE, -2, -1, 0, 1, 2, 10], dtype=torch.float)))

    def test_only_positive(self):
        values = torch.arange(0, 9)
        n_bins = 8
        DB = DistanceBinning(n_bins, n_fixed=5)
        ixs, bins = DB(values.to(torch.long))

        self.assertTrue(bins[ixs].allclose(torch.tensor([0, 1, 2, 3, 4, 6, 6, 8, 8], dtype=torch.float)))

    def test_continuous_distances(self):
        values = torch.tensor([0.1, 1.2, 2.3, 4.5, 4.5, 5.6, 6.7, 7.8, 8.9])
        n_bins = 8
        DB = DistanceBinning(n_bins, n_fixed=5)
        ixs, bins = DB(values)

        self.assertEqual(bins[0], 0.1)
        self.assertEqual(bins[-1], 8.9)
