import os
import glob
import numpy as np
import cv2

from core.optics import matrix_K
from core.undistort import build_undistort_map

# =========================================================
# Config
# =========================================================
W, H = 593, 518
fx, fy = 383.418, 382.382
cx, cy = 296.892, 260.068

n_port = np.array([0.0, 0.0, 1.0])
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33
rflat, tglass = 0.02, 0.002

RGB_DIR    = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/feb/rgb_0"
DEPTH_DIR  = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/feb/ud2"
OUTPUT_DIR = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/feb_ud2/rgb_0"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    K, K_inv = matrix_K(fx, fy, cx, cy)
    kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                  mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)

    rgb_paths = sorted(glob.glob(f"{RGB_DIR}/*.JPG"))
    print(f"Found {len(rgb_paths)} images")

    for rgb_path in rgb_paths:
        img_name    = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path  = f"{DEPTH_DIR}/{img_name}.npy"
        output_path = f"{OUTPUT_DIR}/{img_name}.JPG"

        if os.path.exists(output_path):
            print(f"Skip (exists): {img_name}")
            continue
        if not os.path.exists(depth_path):
            print(f"No depth: {img_name} → skip")
            continue

        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            print(f"Failed to read: {rgb_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth_map = np.load(depth_path).astype(np.float64)
        if depth_map.shape[:2] != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_map = np.clip(depth_map, 1e-3, None)

        map_u, map_v, best_s, Z_ref = build_undistort_map(
            fx, fy, cx, cy, depth_map, kwargs, W, H, K, n_port, rflat, tglass, mu_a, mu_g, mu_w
        )

        u_grid, v_grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        disparity = np.sqrt((map_u - u_grid)**2 + (map_v - v_grid)**2)
        rmse = np.sqrt(np.mean(disparity**2))

        map_u_f = np.nan_to_num(map_u, nan=-1).astype(np.float32)
        map_v_f = np.nan_to_num(map_v, nan=-1).astype(np.float32)
        img_corrected = cv2.remap(img_rgb, map_u_f, map_v_f, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        cv2.imwrite(output_path, cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR))
        print(f"Saved: {img_name}.JPG | Z_ref={Z_ref:.3f}, s={best_s:.4f}, RMSE={rmse:.2f}px")

    print("Done.")


if __name__ == "__main__":
    main()