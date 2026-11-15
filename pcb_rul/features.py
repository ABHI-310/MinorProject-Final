"""Build causal features with scaling fitted only on training units."""
import argparse

import numpy as np

if __package__:
    from .config import OBSERVED, OUTPUTS, WINDOWS
else:
    from config import OBSERVED, OUTPUTS, WINDOWS

N_WINDOW_FEATURES = len(WINDOWS) * len(OBSERVED) * 3
N_FEATURES = N_WINDOW_FEATURES + len(OBSERVED) + 7 + 1


def unit_slices(units):
    edges = np.r_[0, np.flatnonzero(np.diff(units)) + 1, len(units)]
    return (slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True))


def unit_features(values, steps):
    values = np.asarray(values, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.float64)
    end = np.arange(1, len(values) + 1)
    zeros = np.zeros((1, values.shape[1]))

    def prefix(array):
        return np.vstack((zeros, np.cumsum(array, axis=0)))

    sums = prefix(values)
    squares = prefix(values**2)
    time_products = prefix(values * steps[:, None])
    columns = []
    for width in WINDOWS:
        start = np.maximum(0, end - width)
        size = (end - start)[:, None]
        count_sum = sums[end] - sums[start]
        mean = count_sum / size
        std = np.sqrt(np.maximum(0, (squares[end] - squares[start]) / size - mean**2))
        center = ((start + end - 1) / 2)[:, None]
        slope = (time_products[end] - time_products[start] - center * count_sum) / np.maximum(
            size * (size**2 - 1) / 12, 1,
        )
        columns.extend((mean, std, slope))
    ewma = np.empty_like(values)
    ewma[0] = values[0]
    for i in range(1, len(values)):
        ewma[i] = 0.05 * values[i] + 0.95 * ewma[i - 1]
    return np.column_stack((*columns, ewma, sums[1:, [0, 1, 2, 3, 4, 5, 9]], steps))


def load_split(split):
    with np.load(OUTPUTS / f"{split}.npz") as source:
        return dict(source)


def build(split, data=None):
    if data is None:
        data = load_split(split)
    path = OUTPUTS / f"{split}_features.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(data["step"]), N_FEATURES))
    for rows in unit_slices(data["unit"]):
        matrix[rows] = unit_features(data["observed"][rows], data["step"][rows])

    if split == "train":
        from sklearn.preprocessing import StandardScaler

        tabular = StandardScaler().fit(matrix)
        sequence = StandardScaler().fit(data["observed"])
        np.savez(OUTPUTS / "scaling.npz", tabular_mean=tabular.mean_, tabular_scale=tabular.scale_,
                 sequence_mean=sequence.mean_, sequence_scale=sequence.scale_)
    with np.load(OUTPUTS / "scaling.npz") as stats:
        mean, scale = stats["tabular_mean"], stats["tabular_scale"]
        for start in range(0, len(matrix), 16_384):
            rows = matrix[start:start + 16_384]
            rows[:] = (rows - mean) / scale
    matrix.flush()
    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for split in ("train", "val"):
        print(f"{split} features: {build(split).shape}", flush=True)
