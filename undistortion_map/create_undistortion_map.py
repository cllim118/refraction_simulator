import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from core.optics import trace_rays, matrix_K
from core.utilities import make_checkerboard, compute_disparity, sweep_s_gt, blue_gray

# =========================================================
# Camera intrinsics
# =========================================================
W, H = 1920, 1080
fx, fy = 918.30, 919.30
cx, cy = 960.60, 538.60

# =========================================================
# Medium
# =========================================================
n_port = np.array([0.0, 0.0, 1.0])
Z0     = 1.0
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33

# =========================================================
# Checkerboard
# =========================================================
board_cols, board_rows = 10, 10
square_size = 0.05

# =========================================================
# Adaptive inverse map
# =========================================================
def make_inverse_map_adaptive(map_x, map_y, H, W):
    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32)
    )

    valid = ~np.isnan(map_x) & ~np.isnan(map_y)

    x_min = np.floor(map_x[valid].min()).astype(int)
    x_max = np.ceil(map_x[valid].max()).astype(int)
    y_min = np.floor(map_y[valid].min()).astype(int)
    y_max = np.ceil(map_y[valid].max()).astype(int)

    out_W = x_max - x_min
    out_H = y_max - y_min
    print(f"Forward map range: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print(f"Output canvas: {out_W} x {out_H}")

    map_x_shifted = map_x - x_min
    map_y_shifted = map_y - y_min

    src   = np.stack([map_x_shifted.flatten(), map_y_shifted.flatten()], axis=1)
    dst_x = u_grid.flatten()
    dst_y = v_grid.flatten()

    out_grid_x, out_grid_y = np.meshgrid(
        np.arange(out_W, dtype=np.float32),
        np.arange(out_H, dtype=np.float32)
    )

    undist_x = griddata(src, dst_x, (out_grid_x, out_grid_y), method='linear').astype(np.float32)
    undist_y = griddata(src, dst_y, (out_grid_x, out_grid_y), method='linear').astype(np.float32)

    cx_new = cx - x_min
    cy_new = cy - y_min

    return undist_x, undist_y, out_W, out_H, cx_new, cy_new


def save_map(undist_x, undist_y, cx_new, cy_new, out_W, out_H, best_s, path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("Undistortion Map", np.stack([undist_x, undist_y], axis=-1))
    fs.write("cx_new", float(cx_new))
    fs.write("cy_new", float(cy_new))
    fs.write("out_W",  int(out_W))
    fs.write("out_H",  int(out_H))
    fs.write("best_s", float(best_s))
    fs.release()
    print(f"Saved: {path}")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    K, K_inv = matrix_K(fx, fy, cx, cy)

    rflat  = 0.02
    tglass = 0.002

    # trace rays
    map_x, map_y, P_world = trace_rays(
        H, W, K, K_inv, n_port,
        rflat, tglass, Z0,
        mu_a, mu_g, mu_w
    )

    # pixel grid
    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=float),
        np.arange(H, dtype=float)
    )

    # best_s
    _, _, best_s, _, best_rmse = sweep_s_gt(map_x, map_y, u_grid, v_grid, cx, cy)
    print(f"best_s: {best_s:.4f}")

    # scaling
    scaled_map_x = (map_x - cx) * best_s + cx
    scaled_map_y = (map_y - cy) * best_s + cy

    # adaptive inverse map
    print("Computing adaptive inverse map...")
    undist_x, undist_y, out_W, out_H, cx_new, cy_new = make_inverse_map_adaptive(
        scaled_map_x, scaled_map_y, H, W
    )
    print(f"cx_new={cx_new:.2f}, cy_new={cy_new:.2f}")

    # checkerboard
    squares        = make_checkerboard(P_world, board_cols, board_rows, square_size)
    img_underwater = squares.astype(np.uint8)

    # undistortion
    img_corrected = cv2.remap(img_underwater, undist_x, undist_y, cv2.INTER_LINEAR)

    # disparity
    disparity        = compute_disparity(map_x,        map_y,        u_grid, v_grid)
    disparity_scaled = compute_disparity(scaled_map_x, scaled_map_y, u_grid, v_grid)

    # visualise
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"rflat={rflat}, tglass={tglass}, best_s={best_s:.4f} | "
                 f"input={W}x{H} → output={out_W}x{out_H}", fontsize=12)

    axes[0].imshow(img_underwater, cmap=blue_gray, vmin=0, vmax=255)
    axes[0].set_title(f"underwater image\n({W}x{H})")
    axes[0].axis('off')

    axes[1].imshow(img_corrected, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"corrected image\n({out_W}x{out_H})")
    axes[1].axis('off')

    im = axes[2].imshow(disparity, cmap='hot', origin='upper')
    axes[2].set_title(f"disparity (raw)\nmax={np.nanmax(disparity):.3f} px")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    im = axes[3].imshow(disparity_scaled, cmap='hot', origin='upper')
    axes[3].set_title(f"disparity (scaled)\nmax={np.nanmax(disparity_scaled):.3f} px")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.show()

    save_map(undist_x, undist_y, cx_new, cy_new, out_W, out_H, best_s,
             f"undistortion_map_cam1_s{best_s:.3f}.yaml")