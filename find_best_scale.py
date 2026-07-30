import os
import sys
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.optics import matrix_K
from core.undistort import compute_housing_geometry, forward_map
from core.utilities import sweep_s_gt


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_optimal_scale(Z0, W, H, fx, fy, cx, cy, n_port, rflat, tglass, mu_a, mu_g, mu_w):
    K, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w)
    map_x, map_y = forward_map(P2, ray_water, Z0, fx, fy, cx, cy)
    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32)
    )
    s_values, rmse_values, best_s, best_count, best_rmse = sweep_s_gt(
        map_x, map_y, u_grid, v_grid, cx, cy
    )
    print(f"Z0={Z0}")
    print(f"  optimal scale (ZOOM): {best_s:.4f}")
    print(f"  RMSE at optimal scale: {best_rmse:.4f} px")
    print(f"  pixels under 1px error: {best_count}")
    return map_x, map_y, s_values, rmse_values, best_s, best_rmse


def find_in_bounds_scale(map_x, map_y, s_values, rmse_values, W, H, cx, cy):
    """Among the scales already swept by sweep_s_gt, find the one with the
    lowest RMSE whose scaled map stays fully within [0, W-1] x [0, H-1]
    (i.e. does not push any pixel outside the original image bounds)."""
    dx0 = map_x - cx
    dy0 = map_y - cy

    best_s_in_bounds = None
    best_rmse_in_bounds = np.inf

    for s, rmse in zip(s_values, rmse_values):
        scaled_x = dx0 * s + cx
        scaled_y = dy0 * s + cy

        in_bounds = (
            (scaled_x >= 0) & (scaled_x <= W - 1) &
            (scaled_y >= 0) & (scaled_y <= H - 1)
        )
        # every valid (non-NaN) pixel must map inside the frame
        valid = ~np.isnan(scaled_x) & ~np.isnan(scaled_y)
        fully_in_bounds = np.all(in_bounds[valid])

        if fully_in_bounds and rmse < best_rmse_in_bounds:
            best_rmse_in_bounds = rmse
            best_s_in_bounds = s

    return best_s_in_bounds, best_rmse_in_bounds


def main():
    parser = argparse.ArgumentParser(description="Find optimal correction scale from a YAML config")
    parser.add_argument("config", help="Path to config YAML file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    cam = cfg["camera"]
    W, H = cam["W"], cam["H"]
    fx, fy = cam["fx"], cam["fy"]
    cx, cy = cam["cx"], cam["cy"]

    hs = cfg["housing"]
    n_port = np.array(hs["n_port"])
    mu_a, mu_g, mu_w = hs["mu_a"], hs["mu_g"], hs["mu_w"]
    rflat, tglass = hs["rflat"], hs["tglass"]

    Z0_FIXED = cfg["correction"]["z0_fixed"]

    map_x, map_y, s_values, rmse_values, best_s, best_rmse = find_optimal_scale(
        Z0_FIXED, W, H, fx, fy, cx, cy, n_port, rflat, tglass, mu_a, mu_g, mu_w
    )
    print(f"\nBest ZOOM (unconstrained) = {best_s:.4f}")

    best_s_in_bounds, best_rmse_in_bounds = find_in_bounds_scale(
        map_x, map_y, s_values, rmse_values, W, H, cx, cy
    )

    if best_s_in_bounds is not None:
        print(f"\nBest ZOOM (stays within {W}x{H}) = {best_s_in_bounds:.4f}")
        print(f"  RMSE at this scale: {best_rmse_in_bounds:.4f} px")
    else:
        print(f"\nNo scale in the swept range keeps the image fully within {W}x{H}")


if __name__ == "__main__":
    main()