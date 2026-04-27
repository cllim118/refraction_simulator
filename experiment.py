import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from core.optics import trace_rays, matrix_K
from core.utilities import blue_gray, make_checkerboard, compute_disparity, sweep_s_gt

# =========================================================
# Camera intrinsics
# =========================================================
W, H = 500, 500
fx, fy = 700, 700
cx, cy = W / 2.0, H / 2.0

# =========================================================
# Medium
# =========================================================
n_port = np.array([0.0, 0.0, 1.0])
Z0 = 1.0
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33

# =========================================================
# Checkerboard
# =========================================================
board_cols, board_rows = 10, 10
square_size = 0.05

# =========================================================
# Experiment
# =========================================================
def run_experiment(param_values, mode="rflat"):

    n = len(param_values)
   
    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=float),
        np.arange(H, dtype=float)
    )
    
    fig = plt.figure(figsize=(5*n*2, 12))
    gs = GridSpec(3, n*2, figure=fig)

    K, K_inv = matrix_K(fx, fy, cx, cy)

    for i, p in enumerate(param_values):

        if mode == "rflat":
            map_x, map_y, P_world = trace_rays(H, W, K, K_inv, n_port, p, 0.0, Z0, mu_a, mu_g, mu_w)
            title = f"rflat={p}"
        else:
            map_x, map_y, P_world = trace_rays(H, W, K, K_inv, n_port, 0.0, p, Z0, mu_a, mu_g, mu_w)
            title = f"tglass={p}"

        squares = make_checkerboard(P_world, board_cols, board_rows, square_size)
        s_values, errors, best_s, best_error = sweep_s_gt(map_x, map_y, u_grid, v_grid, cx, cy)

        # =====================================================
        # Row 1: underwater image | displacement magnitude
        # =====================================================
        ax = fig.add_subplot(gs[0, i*2])
        ax.scatter(u_grid, v_grid, c=squares, cmap=blue_gray, vmin=0, vmax=255, s=0.1)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title}\nunderwater image")

        disparity = compute_disparity(map_x, map_y, u_grid, v_grid)

        ax = fig.add_subplot(gs[0, i*2+1])
        im = ax.imshow(disparity, cmap='hot', origin='upper')
        ax.set_title(f"pixel displacement\nmax={np.nanmax(disparity):.3f} px")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # =====================================================
        # Row 2: scaled in-air image | displacement magnitude (scaled)
        # =====================================================
        scaled_map_x = (map_x - cx) * best_s + cx
        scaled_map_y = (map_y - cy) * best_s + cy

        ax = fig.add_subplot(gs[1, i*2])
        ax.scatter(scaled_map_x, scaled_map_y, c=squares, cmap='gray', vmin=0, vmax=255, s=0.1)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"scaled in-air image\n(s={best_s:.3f})")

        scaled_disparity = compute_disparity(scaled_map_x, scaled_map_y, u_grid, v_grid)

        ax = fig.add_subplot(gs[1, i*2+1])
        im = ax.imshow(scaled_disparity, cmap='hot', origin='upper')
        ax.set_title(f"pixel displacement (scaled)\nmax={np.nanmax(scaled_disparity):.3f} px")
        fig.colorbar(im, ax=ax, fraction=0.046)

        # =====================================================
        # Row 3: GT-based scale search
        # =====================================================
        ax = fig.add_subplot(gs[2, i*2:i*2+2])
        ax.plot(s_values, errors, linewidth=2)
        ax.axvline(best_s, color='r', linestyle='--', label=f"best s={best_s:.3f}")
        ax.set_title(f"Optimal s={best_s:.3f} (Error: {best_error:.3f})")
        ax.set_xlabel("s")
        ax.set_ylabel("RMSE")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# =========================================================
# Run Experiment
# =========================================================
rflat_values = [0.01, 0.02, 0.05]
tglass_values = [0.01, 0.015, 0.02]

run_experiment(rflat_values,  mode="rflat")
run_experiment(tglass_values, mode="tglass")