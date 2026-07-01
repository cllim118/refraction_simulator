import os
import glob
import numpy as np
import cv2

from core.optics import matrix_K
from core.undistort_depthterm import compute_housing_geometry, build_undistort_map_depthterm

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
DEPTH_DIR  = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/feb/unidepth"
OUTPUT_DIR = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/feb_ud2_z/rgb_0"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    K, K_inv = matrix_K(fx, fy, cx, cy)

    # housing geometry 
    P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass,
                                               mu_a, mu_g, mu_w)

    rgb_paths = sorted(glob.glob(f"{RGB_DIR}/*.JPG"))
    print(f"Found {len(rgb_paths)} images")

    for rgb_path in rgb_paths:
        img_name    = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path  = f"{DEPTH_DIR}/{img_name}.npy"
        output_path = f"{OUTPUT_DIR}/{img_name}.JPG"

        if os.path.exists(output_path):
            continue
        if not os.path.exists(depth_path):
            print(f"No depth: {img_name} → skip")
            continue

        img_bgr = cv2.imread(rgb_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth_map = np.load(depth_path).astype(np.float32)
        if depth_map.shape[:2] != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
        depth_map = np.clip(depth_map, 1e-3, None)

        undist_x, undist_y = build_undistort_map_depthterm(
            fx, fy, cx, cy, depth_map, P2, ray_water, W, H
        )

        undist_x_f = np.nan_to_num(undist_x, nan=-1).astype(np.float32)
        undist_y_f = np.nan_to_num(undist_y, nan=-1).astype(np.float32)

        img_corrected = cv2.remap(img_rgb, undist_x_f, undist_y_f, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        cv2.imwrite(output_path, cv2.cvtColor(img_corrected, cv2.COLOR_RGB2BGR))
        print(f"Saved: {img_name}.JPG")

    print("Done.")


if __name__ == "__main__":
    main()