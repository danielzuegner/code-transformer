import math

import numpy as np


class ExponentialBinning:
    """
    Parametrized exponential function that is fixed by f(0) = 0 and f(max(x)) = max(x).
    This leaves one degree of freedom which is expressed in terms of the growth factor that indicates by how much
    bigger the next bin will be. E.g., a growth factor of 1.3 means that the next bin will be 1.3 times bigger than
    the current one.
    """

    def __init__(self, growth_factor=1.3):
        """
        :param growth_factor: Indicates the desired proportion between bin sizes
        """
        self.growth_factor = growth_factor

    def __call__(self, x, n_bins):
        """
        Calculates exponentially growing bins
        :param x: evenly distributed bin boundaries
        :param n_bins: number of bins
        :return: exponentially growing bins
        """
        max_x = max(x)
        alpha = max_x / (self.growth_factor ** n_bins - 1)
        return alpha * np.power((max_x + alpha) / alpha, x / max_x) - alpha


class EqualBinning:

    def __call__(self, x, n_bins):
        return x


def calculate_bins(x, n_bins, n_fixed, hist_func, trans_func=EqualBinning()):
    """
    :param x: the values to bin
    :param n_bins: number of bins
    :param n_fixed: how many discrete values around 0 should have their own bin. If x has both negative and positive
        values, this will lead to a zero-centered interval. Otherwise, just the first n_fixed numbers will have their
        own bin. E.g., n_fixed = 5 will force bins -2, -1, 0, 1, 2
    :param hist_func: the type of histogram to use. One of hist_by_area or hist_by_number
    :param trans_func: how the bins should be transformed. One of EqualBinning or ExponentialBinning
    :return: n_bins bins for x with fixed bins according to n_fixed and the remaining values being distributed
        according to trans_func
    """
    if n_bins <= n_fixed:
        # Only few unique values => resorting to regular binning
        return hist_func(x, n_bins-1)
    if min(x) < 0:
        # Two-sided binning (negative and positive values)
        lower_end = -math.floor(n_fixed / 2)# - 1
        upper_end = math.ceil(n_fixed / 2)
        fixed_bins = range(lower_end, upper_end)
        # yields a zero-centered list of bin boundaries of size n_fixed + 1. The +1 is necessary as bins are intervals
        # between bin boundaries

        x_lower = x[np.where(x < lower_end)]  # the remaining negative values
        x_upper = x[np.where(x >= upper_end)]  # the remaining positive values

        # number of bins to use for the remaining values. Proportionally distributed among positive and negative values
        n_bins_lower = math.floor(len(x_lower) / (len(x_lower) + len(x_upper)) * (n_bins - n_fixed))
        n_bins_upper = math.ceil(len(x_upper) / (len(x_lower) + len(x_upper)) * (n_bins - n_fixed))

        if n_bins_lower > 1:
            bins_lower = -hist_func(-x_lower, n_bins_lower, trans_func)[:0:-1]
            # exclude biggest value, as it is already accounted by the fixed bins
        elif n_bins_lower == 1:
            bins_lower = [min(x_lower)]
        else:
            bins_lower = []

        if n_bins_upper > 1:
            bins_upper = hist_func(x_upper, n_bins_upper, trans_func)[1:]
            # exclude smallest value, as it is already accounted by the fixed bins
        elif n_bins_upper == 1:
            bins_upper = [max(x_upper)]
        else:
            bins_upper = []

        assert len(bins_lower) + len(fixed_bins) + len(
            bins_upper) == n_bins, f"Calculated wrong amount of bin edge points. Expected {n_bins} got {len(bins_lower) + len(fixed_bins) + len(bins_upper)}"

        bins = np.concatenate((bins_lower, fixed_bins, bins_upper))
        assert (sorted(bins) == bins).all(), f"Calculated bins should increase monotonically. Calculated {bins}"
        return bins
    else:
        # One-sided binning (only positive values)
        if n_bins - n_fixed == 1:
            transformed_bins = [x.max().item()]
        else:
            transformed_bins = hist_func(x[np.where(x > n_fixed)], n_bins - n_fixed - 1)
        assert len(transformed_bins) + n_fixed == n_bins, f"Calculated wrong amount of bin edge points. Expected {n_bins} got {len(transformed_bins) + n_fixed}"

        return np.concatenate((range(n_fixed), transformed_bins))


def hist_by_area(x, n_bins, trans_func=EqualBinning()):
    """
    See https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area
    """
    assert n_bins >= 1, f"Cannot calculate binning for less than 1 bin (got {n_bins})"
    assert len(x) > 0, "Cannot calculate binning on empty list"

    pow = 0.5
    dx = np.diff(np.sort(x))
    tmp = np.cumsum(dx ** pow)
    tmp = np.pad(tmp, (1, 0), 'constant')
    max_x = tmp.max()
    bin_samples = np.linspace(0, max_x, n_bins + 1)
    bin_samples = trans_func(bin_samples, n_bins)
    return np.interp(bin_samples,
                     tmp,
                     np.sort(x))


def hist_by_number(x, n_bins, trans_func=EqualBinning()):
    """
    See https://stackoverflow.com/questions/37649342/matplotlib-how-to-make-a-histogram-with-bins-of-equal-area
    """
    assert n_bins >= 2, f"Cannot calculate binning for less than 2 bins (got {n_bins})"
    assert len(x) > 0, "Cannot calculate binning on empty list"

    n_points = len(x)
    bin_samples = np.linspace(0, n_points, n_bins + 1)
    bin_samples = trans_func(bin_samples, n_bins)
    return np.interp(bin_samples,
                     np.arange(n_points),
                     np.sort(x))
