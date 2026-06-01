"""
gmm.py — Gaussian Mixture Model với EM Algorithm (Numpy only)

Tham khảo: Bishop, PRML Chapter 9 — Mixtures of Gaussians

Mô hình:
    p(x) = Σ_k π_k · N(x | μ_k, Σ_k)

EM:
    E-step: γ_nk = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
    M-step: N_k  = Σ_n γ_nk
            μ_k  = (1/N_k) Σ_n γ_nk x_n
            Σ_k  = (1/N_k) Σ_n γ_nk (x_n-μ_k)(x_n-μ_k)^T
            π_k  = N_k / N
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Numerically stable log-PDF of multivariate Gaussian
# ──────────────────────────────────────────────────────────────────────────────

def _log_gaussian_pdf(X, mean, cov):
    """
    Tính log p(x | mean, cov) cho mỗi điểm trong X.

    Args:
        X    : (N, D)
        mean : (D,)
        cov  : (D, D)

    Returns:
        log_p : (N,)
    """
    D = X.shape[1]
    diff = X - mean                              # (N, D)
    cov_inv = np.linalg.inv(cov)
    sign, log_det = np.linalg.slogdet(cov)
    log_norm = -0.5 * (D * np.log(2 * np.pi) + log_det)
    # Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ)
    maha = np.sum((diff @ cov_inv) * diff, axis=1)   # (N,)
    return log_norm - 0.5 * maha


# ──────────────────────────────────────────────────────────────────────────────
# Khởi tạo tham số GMM
# ──────────────────────────────────────────────────────────────────────────────

def init_gmm(X, k, seed=0):
    """
    Khởi tạo (π, μ, Σ) bằng cách chọn k điểm ngẫu nhiên làm mean ban đầu.

    Returns:
        pis   : (k,)     — mixing coefficients đồng đều 1/k
        means : (k, D)   — mean ban đầu
        covs  : (k, D, D) — identity matrix
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    pis = np.ones(k) / k
    idx = rng.choice(N, k, replace=False)
    means = X[idx].copy().astype(float)
    covs = np.array([np.eye(D) for _ in range(k)])
    return pis, means, covs


# ──────────────────────────────────────────────────────────────────────────────
# E-step: tính responsibility
# ──────────────────────────────────────────────────────────────────────────────

def e_step(X, pis, means, covs):
    """
    E-step: tính γ_nk = p(z_n=k | x_n, θ) bằng log-sum-exp trick.

    Returns:
        gamma : (N, k)  — responsibility của mỗi điểm với mỗi cụm
    """
    k = len(pis)
    N = X.shape[0]

    # Ma trận log-responsibility chưa normalize: (N, k)
    log_r = np.zeros((N, k))
    for j in range(k):
        log_r[:, j] = np.log(pis[j] + 1e-300) + _log_gaussian_pdf(X, means[j], covs[j])

    # Log-sum-exp trick để tránh overflow/underflow
    log_r_max = log_r.max(axis=1, keepdims=True)   # (N, 1)
    log_r -= log_r_max
    gamma = np.exp(log_r)
    gamma /= gamma.sum(axis=1, keepdims=True)        # normalize theo hàng

    return gamma


# ──────────────────────────────────────────────────────────────────────────────
# M-step: cập nhật tham số
# ──────────────────────────────────────────────────────────────────────────────

def m_step(X, gamma, reg=1e-6):
    """
    M-step: cập nhật π_k, μ_k, Σ_k từ responsibility.

    Args:
        gamma : (N, k)
        reg   : regularization thêm vào đường chéo Σ_k để tránh singular

    Returns:
        pis   : (k,)
        means : (k, D)
        covs  : (k, D, D)
    """
    N, D = X.shape
    k = gamma.shape[1]

    Nk = gamma.sum(axis=0)              # (k,)  — số điểm hiệu dụng mỗi cụm

    # Cập nhật mixing coefficients
    pis = Nk / N                        # (k,)

    # Cập nhật means
    means = (gamma.T @ X) / Nk[:, np.newaxis]   # (k, D)

    # Cập nhật covariance matrices
    covs = np.zeros((k, D, D))
    for j in range(k):
        diff = X - means[j]             # (N, D)
        # Σ_k = (1/N_k) Σ_n γ_nk (x_n - μ_k)(x_n - μ_k)^T
        weighted = gamma[:, j:j+1] * diff   # (N, D)
        covs[j] = (weighted.T @ diff) / Nk[j]
        covs[j] += reg * np.eye(D)     # regularization tránh matrix suy biến

    return pis, means, covs


# ──────────────────────────────────────────────────────────────────────────────
# Log-likelihood
# ──────────────────────────────────────────────────────────────────────────────

def compute_log_likelihood(X, pis, means, covs):
    """
    Tính log p(X | θ) = Σ_n log [Σ_k π_k N(x_n | μ_k, Σ_k)].

    Dùng log-sum-exp để tránh underflow.
    """
    k = len(pis)
    N = X.shape[0]

    log_comp = np.zeros((N, k))         # log(π_k N(x_n|μ_k,Σ_k))
    for j in range(k):
        log_comp[:, j] = np.log(pis[j] + 1e-300) + _log_gaussian_pdf(X, means[j], covs[j])

    # log-sum-exp theo cột (axis=1)
    log_comp_max = log_comp.max(axis=1)
    log_likelihood = log_comp_max + np.log(
        np.exp(log_comp - log_comp_max[:, np.newaxis]).sum(axis=1)
    )
    return log_likelihood.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Full EM training
# ──────────────────────────────────────────────────────────────────────────────

def gmm_fit(X, k, max_iter=200, tol=1e-4, seed=0):
    """
    Train Gaussian Mixture Model bằng EM algorithm.

    Args:
        X        : (N, D) — dataset
        k        : số components
        max_iter : số vòng lặp tối đa
        tol      : ngưỡng hội tụ (thay đổi log-likelihood)
        seed     : seed khởi tạo

    Returns:
        pis      : (k,)      — mixing coefficients
        means    : (k, D)    — mean của mỗi Gaussian
        covs     : (k, D, D) — covariance của mỗi Gaussian
        gamma    : (N, k)    — responsibility cuối
        labels   : (N,)      — nhãn cứng (argmax gamma)
        n_iter   : số vòng đã chạy
        log_lh   : log-likelihood cuối
    """
    X = np.asarray(X, dtype=float)
    pis, means, covs = init_gmm(X, k, seed=seed)
    log_lh = -np.inf

    for i in range(1, max_iter + 1):
        gamma = e_step(X, pis, means, covs)
        pis, means, covs = m_step(X, gamma)
        new_ll = compute_log_likelihood(X, pis, means, covs)

        if abs(new_ll - log_lh) < tol:
            break
        log_lh = new_ll

    labels = np.argmax(gamma, axis=1)
    return pis, means, covs, gamma, labels, i, log_lh
