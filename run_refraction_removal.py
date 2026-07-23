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
fx, fy = 915.65, 914.82
cx, cy = 966.025, 529.764

n_port = np.array([0.0, 0.0, 1.0])
mu_a, mu_g, mu_w = 1.0, 1.47, 1.33
rflat, tglass = 0.02, 0.002

RGB_DIR    = "/home/chelim/Bommie/jul2026_malaysia/p1/s02/syncd/p1_s02_C2"
OUTPUT_DIR = "/home/chelim/Bommie/jul2026_malaysia/p1/s02/corrected/p1_s02_C2_c"
DEPTH_DIR  = None
Z0_FIXED   = 1
ZOOM       = None
STEP_SIZE = 1


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

    IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
                  "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]
    rgb_paths = sorted(
        p for ext in IMAGE_EXTS for p in glob.glob(f"{RGB_DIR}/{ext}")
    )
    rgb_paths = rgb_paths[::STEP_SIZE]

    print(f"Found {len(rgb_paths)} images (step={STEP_SIZE}), method={METHOD}")

    precomputed_map = None
    if DEPTH_DIR is None:
        print(f"Single depth mode (Z0={Z0_FIXED}) — computing undistortion map once")
        if METHOD == "closed_form":
            undist_x, undist_y = build_undistort_map_closed_form(
                P2, ray_water, Z0_FIXED, fx, fy, cx, cy, H, W, zoom=ZOOM
            )
        else:
            undist_x, undist_y = build_undistort_map_newton(
                fx, fy, cx, cy, Z0_FIXED, kwargs, W, H, zoom=ZOOM or 1.4
            )
        precomputed_map = (np.nan_to_num(undist_x, nan=-1),
                            np.nan_to_num(undist_y, nan=-1))

    for rgb_path in rgb_paths:
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        ext  = os.path.splitext(rgb_path)[1]
        out_path = f"{OUTPUT_DIR}/{name}{ext}"
        if os.path.exists(out_path):
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            continue

        if precomputed_map is not None:
            undist_x, undist_y = precomputed_map
        else:
            depth = load_depth(name)
            if depth is None:
                print(f"No depth: {name} → skip")
                continue

            if METHOD == "closed_form":
                ux, uy = build_undistort_map_closed_form(
                    P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=ZOOM
                )
            else:
                ux, uy = build_undistort_map_newton(
                    fx, fy, cx, cy, depth, kwargs, W, H, zoom=ZOOM or 1.4
                )
            undist_x = np.nan_to_num(ux, nan=-1)
            undist_y = np.nan_to_num(uy, nan=-1)

        corrected = cv2.remap(img, undist_x, undist_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(out_path, corrected)
        print(f"Saved: {name}{ext}")

    print("Done.")

if __name__ == "__main__":
    main()