import os, glob
import numpy as np
import cv2

from core.optics import matrix_K
from core.undistort import compute_housing_geometry, build_undistort_map_closed_form
from core.undistort_newton import build_undistort_map_newton

# =========================================================
# Config
# =========================================================
METHOD = "newton"   # "closed_form" or "newton"

# W, H = 593, 518
# fx, fy = 383.418 / 1.33, 382.382 / 1.33
# cx, cy = 296.892, 260.068

W, H = 1920, 1080
fx, fy = 912.43292772, 911.10108288
cx, cy = 966.24235151, 534.46118474

n_port = np.array([0.0, 0.0, 1.0])
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33
rflat, tglass = 0.02, 0.002

RGB_DIR    = "/home/chelim/hpc/VSLAM-LAB-Benchmark/MALAYSIA2/p01_left/rgb_0"
DEPTH_DIR  = None
Z0_FIXED   = 1.5
ZOOM       = None  
OUTPUT_DIR = "/home/chelim/hpc/VSLAM-LAB-Benchmark/MALAYSIA2/p01_left_corr/rgb_0"


def load_depth(name):
    if DEPTH_DIR is None:
        return Z0_FIXED
    depth_path = f"{DEPTH_DIR}/{name}.npy"
    if not os.path.exists(depth_path):
        return None
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth, 1e-3, None)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    K, K_inv = matrix_K(fx, fy, cx, cy)

    if METHOD == "closed_form":
        P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w)
    else:
        kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                      mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)

    rgb_paths = sorted(glob.glob(f"{RGB_DIR}/*.JPG"))
    print(f"Found {len(rgb_paths)} images, method={METHOD}")

    for rgb_path in rgb_paths:
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        out_path = f"{OUTPUT_DIR}/{name}.JPG"
        if os.path.exists(out_path):
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            continue

        depth = load_depth(name)
        if depth is None:
            print(f"No depth: {name} → skip")
            continue

        if METHOD == "closed_form":
            undist_x, undist_y = build_undistort_map_closed_form(
                P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=ZOOM
            )
        else:
            undist_x, undist_y = build_undistort_map_newton(
                fx, fy, cx, cy, depth, kwargs, W, H, zoom=ZOOM or 1.4
            )

        undist_x = np.nan_to_num(undist_x, nan=-1)
        undist_y = np.nan_to_num(undist_y, nan=-1)

        corrected = cv2.remap(img, undist_x, undist_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(out_path, corrected)
        print(f"Saved: {name}.JPG")

    print("Done.")


if __name__ == "__main__":
    main()