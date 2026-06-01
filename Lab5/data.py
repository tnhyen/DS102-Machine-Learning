import numpy as np


def make_gaussian_mixture(means, covs, sizes, seed=42):
    """
    Tạo dataset gồm nhiều cụm Gaussian.

    Args:
        means : list of (D,) array — trung bình mỗi cụm
        covs  : list of (D,D) array — covariance mỗi cụm
        sizes : list of int — số điểm mỗi cụm
        seed  : random seed để tái lập kết quả

    Returns:
        X      : (N, D) ndarray
        labels : (N,)  ndarray — nhãn thực (0, 1, 2, ...)
    """
    rng = np.random.default_rng(seed)
    parts = [rng.multivariate_normal(m, c, n) for m, c, n in zip(means, covs, sizes)]
    X = np.vstack(parts)
    labels = np.concatenate([np.full(n, k) for k, n in enumerate(sizes)])
    return X, labels
