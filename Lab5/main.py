import matplotlib
matplotlib.use('Agg')          # non-interactive — không cần màn hình
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np

from data import make_gaussian_mixture
from kmeans import kmeans_fit
from gmm import gmm_fit
from visualization import (plot_before_after, plot_multi_seeds,
                            plot_gmm_result, plot_background_filter)

# Auto-save: mỗi lần plt.show() gọi → lưu PNG thay vì mở cửa sổ
_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
_fig_counter = [0]
def _auto_save_show():
    _fig_counter[0] += 1
    path = os.path.join(_OUT_DIR, f"output_fig_{_fig_counter[0]:02d}.png")
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {os.path.basename(path)}")
plt.show = _auto_save_show

K = 3
SEEDS = [0, 1, 7, 13, 99, 123]

# ──────────────────────────────────────────────────────────────────────────────
# Assignment 1 — Balanced clusters, Σ = I
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ASSIGNMENT 1 — Balanced (200/200/200), Σ = I")
print("=" * 60)

I2 = np.eye(2)
means = [np.array([2, 2]), np.array([8, 3]), np.array([3, 6])]

X1, true1 = make_gaussian_mixture(
    means=means,
    covs=[I2, I2, I2],
    sizes=[200, 200, 200],
    seed=42,
)

centroids1, labels1, n_iter1, loss1 = kmeans_fit(X1, K, seed=0)

print(f"Centroids cuối:\n{centroids1.round(3)}")
print(f"Số vòng lặp  : {n_iter1}")
print(f"Inertia (J)  : {loss1:.2f}\n")

plot_before_after(X1, true1, labels1, centroids1,
                  title="Assignment 1 — Balanced (200/200/200), Σ = I")

plot_multi_seeds(X1, K, kmeans_fit, SEEDS,
                 title="Assignment 1 — Ảnh hưởng của seed khởi tạo")

print("""
Nhận xét (Assignment 1):
- 3 cụm tách biệt rõ và đều nhau → K-Means hội tụ đúng với hầu hết seed.
- Nếu 2 centroid ban đầu cùng rơi vào một cụm → hội tụ cục bộ (local optimum):
  một cụm bị chia đôi, cụm còn lại bị bỏ sót.
- Giải pháp: chạy nhiều lần với seed khác nhau, chọn J nhỏ nhất (multiple restarts).
""")

# ──────────────────────────────────────────────────────────────────────────────
# Assignment 2 — Imbalanced clusters (1200 / 200 / 1000), Σ = I
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ASSIGNMENT 2 — Imbalanced (1200/200/1000), Σ = I")
print("=" * 60)

X2, true2 = make_gaussian_mixture(
    means=means,
    covs=[I2, I2, I2],
    sizes=[1200, 200, 1000],
    seed=42,
)

centroids2, labels2, n_iter2, loss2 = kmeans_fit(X2, K, seed=0)

print(f"Centroids cuối:\n{centroids2.round(3)}")
print(f"Số vòng lặp  : {n_iter2}")
print(f"Inertia (J)  : {loss2:.2f}\n")

plot_before_after(X2, true2, labels2, centroids2,
                  title="Assignment 2 — Imbalanced (1200/200/1000), Σ = I")

plot_multi_seeds(X2, K, kmeans_fit, SEEDS,
                 title="Assignment 2 — Ảnh hưởng của seed (imbalanced)")

print("""
Nhận xét (Assignment 2):
- Cụm nhỏ (200 điểm quanh (8,3)) dễ bị centroid của cụm lớn "lấn vào" khi seed xấu.
- K-Means chỉ tối thiểu hoá khoảng cách, không mô hình hoá kích thước cụm →
  ranh giới phân chia đặt ở trung điểm 2 centroid, không phải nơi tự nhiên nhất.
- Kết quả kém ổn định hơn Assignment 1 khi đổi seed.
- Gaussian Mixture Model (với mixing coefficient π_k) xử lý trường hợp này tốt hơn.
""")

# ──────────────────────────────────────────────────────────────────────────────
# Assignment 3 — Anisotropic covariance (Σ₂ = diag(10, 1))
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ASSIGNMENT 3 — Anisotropic: cụm 3 ~ N((3,6), diag(10,1))")
print("=" * 60)

Sigma2 = np.array([[10, 0], [0, 1]])   # trải rộng theo trục x

X3, true3 = make_gaussian_mixture(
    means=means,
    covs=[I2, I2, Sigma2],
    sizes=[200, 200, 200],
    seed=42,
)

centroids3, labels3, n_iter3, loss3 = kmeans_fit(X3, K, seed=0)

print(f"Centroids cuối:\n{centroids3.round(3)}")
print(f"Số vòng lặp  : {n_iter3}")
print(f"Inertia (J)  : {loss3:.2f}\n")

plot_before_after(X3, true3, labels3, centroids3,
                  title="Assignment 3 — Cụm 3: N((3,6), diag(10,1))")

plot_multi_seeds(X3, K, kmeans_fit, SEEDS,
                 title="Assignment 3 — Ảnh hưởng của seed (anisotropic)")

print("""
Nhận xét (Assignment 3):
- Cụm 3 trải rộng theo trục x (phương sai=10) → chồng lấp đáng kể với cụm 1.
- K-Means phân chia theo hình cầu (Euclidean) → không thích nghi hình elip →
  nhiều điểm của cụm 3 bị gán nhầm vào cụm 1 hoặc cụm 2.
- Kết quả không ổn định và sai lệch nhiều hơn 2 assignment trước.
- Cần dùng GMM với Σ_k riêng biệt cho từng cụm để xử lý trường hợp này.
""")

# ══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN MIXTURE MODEL — Assignment 1
# Implement GMM bằng Numpy, train với EM
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("GMM ASSIGNMENT 1 — Gaussian Mixture Model với EM")
print("═" * 60)

# ── GMM trên dataset balanced (giống K-Means A1) ─────────────────────────────
print("\n[GMM A1 - Dataset 1] Balanced 200/200/200, Σ = I")
print("-" * 50)

pis1, means1_gmm, covs1_gmm, gamma1, labels1_gmm, n_iter1_gmm, ll1 = gmm_fit(
    X1, K, seed=0
)

print(f"Số vòng lặp EM : {n_iter1_gmm}")
print(f"Log-likelihood  : {ll1:.2f}")
print(f"Mixing coeff π  : {pis1.round(4)}")
print("Mean μ_k:")
for j, m in enumerate(means1_gmm):
    print(f"  Cụm {j+1}: {m.round(3)}")
print("Covariance Σ_k (đường chéo chính):")
for j, c in enumerate(covs1_gmm):
    print(f"  Cụm {j+1}: diag = {np.diag(c).round(3)}")

plot_gmm_result(X1, true1, labels1_gmm, means1_gmm, covs1_gmm,
                title="GMM A1 — Balanced (200/200/200), Σ = I")

# ── GMM trên dataset anisotropic (giống K-Means A3) ──────────────────────────
print("\n[GMM A1 - Dataset 2] Anisotropic — cụm 3 ~ N((3,6), diag(10,1))")
print("-" * 50)

pis3, means3_gmm, covs3_gmm, gamma3, labels3_gmm, n_iter3_gmm, ll3 = gmm_fit(
    X3, K, seed=0
)

print(f"Số vòng lặp EM : {n_iter3_gmm}")
print(f"Log-likelihood  : {ll3:.2f}")
print(f"Mixing coeff π  : {pis3.round(4)}")
print("Mean μ_k:")
for j, m in enumerate(means3_gmm):
    print(f"  Cụm {j+1}: {m.round(3)}")
print("Covariance Σ_k (đường chéo chính):")
for j, c in enumerate(covs3_gmm):
    print(f"  Cụm {j+1}: diag = {np.diag(c).round(3)}")

plot_gmm_result(X3, true3, labels3_gmm, means3_gmm, covs3_gmm,
                title="GMM A1 — Anisotropic: cụm 3 ~ N((3,6), diag(10,1))")

print("""
Nhận xét (GMM Assignment 1):
- Dataset balanced: GMM cho kết quả tương đương K-Means khi covariance đồng đều.
  Σ_k học được ≈ I (identity), π_k ≈ 1/3 — phù hợp với dữ liệu.
- Dataset anisotropic (Σ_2 = diag(10,1)):
  GMM học được Σ_k phù hợp với hình dạng elip của cụm 3 (phương sai lớn theo trục x).
  → Kết quả phân cụm chính xác hơn K-Means đáng kể.
- K-Means dùng khoảng cách Euclidean (giả định cụm tròn) →
  GMM dùng covariance riêng biệt cho mỗi cụm → linh hoạt hơn nhiều.
""")

# ══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN MIXTURE MODEL — Assignment 2
# Dùng GMM để lọc background của ảnh cow.jpg
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print("GMM ASSIGNMENT 2 — Background Filtering (cow.jpg)")
print("═" * 60)

img_path = os.path.join(os.path.dirname(__file__), "cow.jpg")
img = mpimg.imread(img_path)

# Đảm bảo ảnh là uint8 [0, 255]
if img.dtype != np.uint8:
    img = (img * 255).astype(np.uint8)

H, W, C = img.shape
print(f"Kích thước ảnh  : {H} x {W} x {C}")
print(f"Tổng số pixel   : {H * W}")

# Reshape thành (N, 3) và chuẩn hoá về [0, 1] để GMM hội tụ ổn định hơn
pixels = img.reshape(-1, C).astype(float) / 255.0

# Dùng K=3 component: thường ảnh ngoài trời có nhiều vùng màu
K_img = 3
print(f"\nFit GMM với K={K_img} components trên không gian màu RGB...")

pis_img, means_img, covs_img, gamma_img, labels_img, n_iter_img, ll_img = gmm_fit(
    pixels, K_img, max_iter=50, tol=1e-4, seed=42
)

print(f"Số vòng lặp EM  : {n_iter_img}")
print(f"Log-likelihood  : {ll_img:.2f}")
print(f"Mixing coeff π  : {pis_img.round(4)}")
print("Mean màu RGB (μ_k) của mỗi component:")
for j, m in enumerate(means_img):
    print(f"  Component {j+1}: R={m[0]:.3f}  G={m[1]:.3f}  B={m[2]:.3f}  (π={pis_img[j]:.3f})")

# Xác định component "background":
# Thường background chiếm tỷ lệ lớn hoặc có màu đặc trưng (trời/đất).
# Dùng heuristic: component có mixing coefficient lớn nhất là background.
bg_component = np.argmax(pis_img)
print(f"\nComponent background được chọn: {bg_component + 1} (π = {pis_img[bg_component]:.3f})")

# Tạo mask background: pixel nào có responsibility cao nhất thuộc bg_component
background_mask = (labels_img == bg_component).reshape(H, W)

print(f"Số pixel background : {background_mask.sum()} / {H * W} "
      f"({100 * background_mask.mean():.1f}%)")

plot_background_filter(img, background_mask,
                       title=f"GMM Background Filtering — K={K_img} components")

print("""
Nhận xét (GMM Assignment 2):
- GMM phân không gian màu RGB thành K cụm; mỗi cụm học phân phối màu riêng.
- Pixel được gán vào cụm theo responsibility γ_nk — quyết định mềm (soft assignment).
- Heuristic chọn background: component có π_k lớn nhất (chiếm nhiều pixel nhất).
  Trong thực tế, có thể kết hợp thêm thông tin vị trí không gian (spatial GMM).
- GMM phù hợp hơn K-Means cho task này vì các vùng màu ảnh thực thường có
  phân phối elipsoid (không tròn đều) trong không gian RGB.
- Hạn chế: GMM không tận dụng thông tin kết nối láng giềng → đôi khi có
  nhiễu rải rác; có thể cải thiện bằng CRF / morphological post-processing.
""")
