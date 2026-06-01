import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']


def _cat_marker(ax, x, y, size=16, zorder=6):
    """
    Vẽ emoji con mèo 🐱 tại vị trí (x, y) thay cho marker 'X'.
    Dùng ax.text() để render emoji trực tiếp trên plot.
    """
    ax.text(x, y, '🐱', fontsize=size, ha='center', va='center', zorder=zorder)


def plot_before_after(X, true_labels, pred_labels, centroids, title=""):
    """
    Vẽ 2 scatter plot cạnh nhau:
      - Trái : nhãn thực (ground truth)
      - Phải : kết quả K-Means + centroid
    """
    k = len(centroids)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Ground truth
    for j in np.unique(true_labels):
        pts = X[true_labels == j]
        axes[0].scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6,
                        color=COLORS[j % len(COLORS)], label=f"Cụm {j+1}")
    axes[0].set_title("Ground truth")
    axes[0].legend(markerscale=2, fontsize=9)

    # K-Means result
    for j in range(k):
        pts = X[pred_labels == j]
        axes[1].scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6,
                        color=COLORS[j % len(COLORS)], label=f"Cụm {j+1}")
    for cx, cy in centroids:
        _cat_marker(axes[1], cx, cy, size=14, zorder=6)
    axes[1].scatter([], [], marker='$🐱$', s=80, c='black', label='Centroids')
    axes[1].set_title("Kết quả K-Means")
    axes[1].legend(markerscale=2, fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_multi_seeds(X, k, fit_fn, seeds, title=""):
    """
    Chạy K-Means với nhiều seed khác nhau và vẽ kết quả dạng grid.

    Args:
        fit_fn : callable(X, k, seed) → (centroids, labels, n_iter, loss)
    """
    n = len(seeds)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flat

    for ax, seed in zip(axes, seeds):
        c, lbl, n_it, loss = fit_fn(X, k, seed=seed)
        for j in range(k):
            pts = X[lbl == j]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5,
                       color=COLORS[j % len(COLORS)])
        for cx, cy in c:
            _cat_marker(ax, cx, cy, size=11, zorder=6)
        ax.set_title(f"seed={seed} | iter={n_it} | J={loss:.0f}", fontsize=9)

    # Ẩn các subplot thừa
    for ax in list(axes)[len(seeds):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _draw_ellipse(ax, mean, cov, color, n_std=2.0, alpha=0.25):
    """
    Vẽ ellipse thể hiện covariance (n_std lần độ lệch chuẩn).
    """
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean[:2], width=width, height=height,
                  angle=angle, color=color, alpha=alpha, zorder=3)
    ax.add_patch(ell)
    _cat_marker(ax, mean[0], mean[1], size=14, zorder=5)


def plot_gmm_result(X, true_labels, pred_labels, means, covs,
                    title="GMM Result"):
    """
    Vẽ 2 panel: Ground truth | GMM clusters với ellipse covariance.
    """
    k = len(means)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Ground truth
    for j in np.unique(true_labels):
        pts = X[true_labels == j]
        axes[0].scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6,
                        color=COLORS[j % len(COLORS)], label=f"Cụm {j+1}")
    axes[0].set_title("Ground truth", fontsize=11)
    axes[0].legend(markerscale=2, fontsize=9)

    # GMM result with covariance ellipses
    for j in range(k):
        pts = X[pred_labels == j]
        c = COLORS[j % len(COLORS)]
        axes[1].scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.6, color=c,
                        label=f"Cụm {j+1}")
        _draw_ellipse(axes[1], means[j], covs[j], color=c)
    axes[1].set_title("Kết quả GMM (ellipse = 2σ)", fontsize=11)
    axes[1].legend(markerscale=2, fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_background_filter(original_img, background_mask, title="GMM Background Filtering"):
    """
    Hiển thị 3 panel: ảnh gốc | foreground | background.

    Args:
        original_img    : (H, W, 3) uint8
        background_mask : (H, W) bool — True = background pixel
    """
    H, W, _ = original_img.shape
    foreground = original_img.copy()
    background = original_img.copy()

    # Foreground: tô trắng pixel background
    foreground[background_mask] = 255
    # Background: tô trắng pixel foreground
    background[~background_mask] = 255

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_img)
    axes[0].set_title("Ảnh gốc", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(foreground)
    axes[1].set_title("Foreground (vật thể)", fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(background)
    axes[2].set_title("Background", fontsize=11)
    axes[2].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
