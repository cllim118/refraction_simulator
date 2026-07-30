import os
import sys
import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.optics import matrix_K, trace_underwater
from core.undistort import compute_housing_geometry, forward_map, invert_map
from core.undistort_newton import build_undistort_map_newton

# ── Fixed checkerboard parameters ──
BOARD_COLS, BOARD_ROWS, SQUARE_SIZE = 10, 10, 0.1


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_checkerboard(P_world, cols, rows, size, background=127):
    X, Y = P_world[..., 0], P_world[..., 1]
    w, h = cols * size, rows * size
    x0, y0 = -w / 2, -h / 2
    Xl, Yl = X - x0, Y - y0

    mask = (Xl >= 0) & (Xl < w) & (Yl >= 0) & (Yl < h)
    squares = np.full_like(X, background, dtype=np.int32)
    ix = np.floor(Xl[mask] / size).astype(int)
    iy = np.floor(Yl[mask] / size).astype(int)
    squares[mask] = np.where((ix + iy) % 2 == 0, 255, 0)
    return squares


blue_gray = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue_gray', ['black', (0.70, 0.82, 1.0)]
)


def plot_radial_disparity(map_u, map_v, u_grid, v_grid, cx, cy):
    r = np.sqrt((u_grid - cx) ** 2 + (v_grid - cy) ** 2)
    disparity = np.sqrt((map_u - u_grid) ** 2 + (map_v - v_grid) ** 2)

    r_flat, d_flat = r.ravel(), disparity.ravel()
    order = np.argsort(r_flat)
    r_sorted, d_sorted = r_flat[order], d_flat[order]

    bins = np.linspace(0, r_sorted.max(), 51)
    bin_idx = np.digitize(r_sorted, bins) - 1
    bin_means = [d_sorted[bin_idx == i].mean() if np.any(bin_idx == i) else np.nan
                 for i in range(50)]
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(r_flat, d_flat, s=2, alpha=0.1, color='gray', label='per-pixel')
    ax.plot(bin_centers, bin_means, color='crimson', linewidth=2, label='binned mean')
    ax.set_xlabel("radial distance r (px)"); ax.set_ylabel("pixel disparity (px)")
    ax.set_title("Disparity vs radius")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Visualise refraction correction from a YAML config")
    parser.add_argument("config", help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    method = cfg["method"]

    cam = cfg["camera"]
    W, H = cam["W"], cam["H"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]

    hs = cfg["housing"]
    n_port = np.array(hs["n_port"])
    mu_a, mu_g, mu_w = hs["mu_a"], hs["mu_g"], hs["mu_w"]
    rflat, tglass = hs["rflat"], hs["tglass"]

    corr = cfg["correction"]
    Z0 = corr["z0_fixed"]
    ZOOM = corr["zoom"]

    K, K_inv = matrix_K(fx, fy, cx, cy)
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))

    if method == "closed_form":
        P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w)
        fwd_x, fwd_y = forward_map(P2, ray_water, Z0, fx, fy, cx, cy)

        map_x = fwd_x if ZOOM is None else (fwd_x - cx) * ZOOM + cx
        map_y = fwd_y if ZOOM is None else (fwd_y - cy) * ZOOM + cy
        map_u, map_v = invert_map(map_x, map_y, H, W)

    else:  # newton
        kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                      mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)

        P_water = trace_underwater(u_grid, v_grid, K_inv, n_port, rflat, tglass, Z0,
                                    mu_a, mu_g, mu_w)

        fwd_x = fx * P_water[..., 0] / Z0 + cx
        fwd_y = fy * P_water[..., 1] / Z0 + cy
        map_u, map_v = build_undistort_map_newton(fx, fy, cx, cy, Z0, kwargs, W, H, zoom=ZOOM or 1.4)

    disparity = np.sqrt((map_u - u_grid) ** 2 + (map_v - v_grid) ** 2)
    rmse = np.sqrt(np.nanmean(disparity ** 2))
    p95 = np.nanpercentile(disparity, 95)
    print(f"method={method} | Z0={Z0}, zoom={ZOOM}")
    print(f"pixel movement: RMSE={rmse:.2f}px, max={np.nanmax(disparity):.2f}px, p95={p95:.2f}px")

    # ── underwater image (world point -> checkerboard) ──
    P_world_underwater = np.stack([
        (fwd_x - cx) / fx * Z0, (fwd_y - cy) / fy * Z0, np.full((H, W), Z0)
    ], axis=-1)
    img_underwater = make_checkerboard(P_world_underwater, BOARD_COLS, BOARD_ROWS, SQUARE_SIZE).astype(np.uint8)

    # ── remove refraction ──
    map_u_f = np.nan_to_num(map_u, nan=-1).astype(np.float32)
    map_v_f = np.nan_to_num(map_v, nan=-1).astype(np.float32)
    img_corrected = cv2.remap(img_underwater, map_u_f, map_v_f, cv2.INTER_LINEAR)

    # ── visualise ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(f"{method} | rflat={rflat}, tglass={tglass}, Z0={Z0}, zoom={ZOOM}", fontsize=12)
    axes[0].imshow(img_underwater, cmap=blue_gray, vmin=0, vmax=255)
    axes[0].set_title(f"underwater ({W}x{H})"); axes[0].axis('off')
    axes[1].imshow(img_corrected, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("corrected"); axes[1].axis('off')
    im = axes[2].imshow(disparity, cmap='hot', origin='upper')
    axes[2].set_title(f"pixel displacement\nRMSE={rmse:.2f}px")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    plt.tight_layout()

    plot_radial_disparity(map_u, map_v, u_grid, v_grid, cx, cy)
    plt.show()


if __name__ == "__main__":
    main()