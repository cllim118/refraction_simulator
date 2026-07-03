import numpy as np
import cv2
import matplotlib.pyplot as plt

from core.optics import matrix_K, trace_rays
from core.utilities import make_checkerboard, blue_gray
from core.undistort import build_undistort_map

# =========================================================
# Camera intrinsics
# =========================================================
W, H = 593, 518
fx, fy = 383.418/1.33, 382.382/1.33
cx, cy = 296.892, 260.068

n_port = np.array([0.0, 0.0, 1.0])
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33
rflat, tglass = 0.02, 0.002
Z0 = 1.0

board_cols, board_rows = 10, 10
square_size = 0.05


def plot_radial_disparity(map_u, map_v, u_grid, v_grid, cx, cy):
    r = np.sqrt((u_grid - cx)**2 + (v_grid - cy)**2)
    disparity = np.sqrt((map_u - u_grid)**2 + (map_v - v_grid)**2)

    r_flat, d_flat = r.ravel(), disparity.ravel()
    order = np.argsort(r_flat)
    r_sorted, d_sorted = r_flat[order], d_flat[order]

    n_bins = 50
    bins = np.linspace(0, r_sorted.max(), n_bins + 1)
    bin_idx = np.digitize(r_sorted, bins) - 1
    bin_means = [d_sorted[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
                 for i in range(n_bins)]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_flat, d_flat, s=2, alpha=0.1, color='gray', label='per-pixel')
    ax.plot(bin_centers, bin_means, color='crimson', linewidth=2, label='binned mean')
    ax.set_xlabel("radial distance from center r (px)")
    ax.set_ylabel("pixel disparity (px)")
    ax.set_title("Disparity vs radius — refraction distortion pattern")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_map(map_u, map_v, path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("Undistortion Map", np.stack([map_u, map_v], axis=-1))
    fs.write("out_W", int(map_u.shape[1])); fs.write("out_H", int(map_u.shape[0]))
    fs.release()
    print(f"Saved: {path}")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    K, K_inv = matrix_K(fx, fy, cx, cy)
    kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                  mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)   # Z0 없음 (build_undistort_map이 처리)

    map_u, map_v, best_s, Z_ref = build_undistort_map(
        fx, fy, cx, cy, Z0, kwargs, W, H, K, n_port, rflat, tglass, mu_a, mu_g, mu_w
        # best_s=1.4   # ← 하드코드하고 싶으면 이 줄 추가
    )
    print(f"Z_ref={Z_ref:.3f}, best_s={best_s:.4f}")
    print(f"map_u range: [{np.nanmin(map_u):.1f}, {np.nanmax(map_u):.1f}]")
    print(f"map_v range: [{np.nanmin(map_v):.1f}, {np.nanmax(map_v):.1f}]")

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    disparity = np.sqrt((map_u - u_grid)**2 + (map_v - v_grid)**2)
    rmse = np.sqrt(np.mean(disparity**2))
    p95  = np.percentile(disparity, 95)
    print(f"pixel movement: RMSE={rmse:.2f}px, max={disparity.max():.2f}px, p95={p95:.2f}px")

    _, _, P_world = trace_rays(H, W, K, K_inv, n_port, rflat, tglass, Z0, mu_a, mu_g, mu_w)
    img_underwater = make_checkerboard(P_world, board_cols, board_rows, square_size).astype(np.uint8)

    map_u_f = np.nan_to_num(map_u, nan=-1).astype(np.float32)
    map_v_f = np.nan_to_num(map_v, nan=-1).astype(np.float32)
    img_corrected = cv2.remap(img_underwater, map_u_f, map_v_f, cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"Zoom-baked Newton | rflat={rflat}, tglass={tglass}, Z0={Z0}, s={best_s:.2f}", fontsize=12)
    axes[0].imshow(img_underwater, cmap=blue_gray, vmin=0, vmax=255); axes[0].set_title(f"underwater\n({W}x{H})"); axes[0].axis('off')
    axes[1].imshow(img_corrected, cmap='gray', vmin=0, vmax=255); axes[1].set_title("corrected (zoom retained)"); axes[1].axis('off')
    im = axes[2].imshow(disparity, cmap='hot', origin='upper'); axes[2].set_title(f"pixel displacement\nRMSE={rmse:.2f}px")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout(); plt.show()

    plot_radial_disparity(map_u, map_v, u_grid, v_grid, cx, cy)

    save_map(map_u, map_v, f"undist_lizard_Z{Z0}_s{best_s:.2f}_2.yaml")