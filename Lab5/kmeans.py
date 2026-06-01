import numpy as np


def init_centroids(X, k, seed=0):
    """Chọn ngẫu nhiên k điểm trong X làm centroid ban đầu."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), k, replace=False)
    return X[idx].copy()


def e_step(X, centroids):
    """
    E-step: gán mỗi điểm vào centroid gần nhất (khoảng cách Euclidean).

    Returns:
        labels : (N,) — chỉ số cụm được gán
    """
    # (N, 1, D) - (1, k, D) → bình phương khoảng cách (N, k)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.sum(diff ** 2, axis=2)
    return np.argmin(dists, axis=1)


def m_step(X, labels, k):
    """
    M-step: cập nhật centroid = trung bình các điểm trong cùng cụm.

    Returns:
        centroids : (k, D)
    """
    D = X.shape[1]
    centroids = np.zeros((k, D))
    for j in range(k):
        pts = X[labels == j]
        if len(pts) > 0:
            centroids[j] = pts.mean(axis=0)
    return centroids


def compute_inertia(X, labels, centroids):
    """Tính tổng bình phương khoảng cách từ mỗi điểm đến centroid của nó (hàm J)."""
    total = 0.0
    for j in range(len(centroids)):
        pts = X[labels == j]
        if len(pts) > 0:
            total += np.sum((pts - centroids[j]) ** 2)
    return total


def kmeans_fit(X, k, max_iter=300, tol=1e-6, seed=0):
    """
    Chạy K-Means EM đến hội tụ.

    Args:
        X        : (N, D) dataset
        k        : số cụm
        max_iter : số vòng lặp tối đa
        tol      : ngưỡng dừng (shift của centroid)
        seed     : seed khởi tạo centroid

    Returns:
        centroids : (k, D) — centroid cuối
        labels    : (N,)   — nhãn dự đoán
        n_iter    : số vòng lặp đã chạy
        loss      : inertia J cuối cùng
    """
    centroids = init_centroids(X, k, seed=seed)

    for i in range(1, max_iter + 1):
        labels = e_step(X, centroids)
        new_centroids = m_step(X, labels, k)

        # Kiểm tra hội tụ: centroid dịch chuyển tối đa bao nhiêu?
        shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        centroids = new_centroids

        if shift < tol:
            break

    loss = compute_inertia(X, labels, centroids)
    return centroids, labels, i, loss
