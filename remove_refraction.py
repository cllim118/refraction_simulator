import os
import sys
import glob
import argparse
import yaml
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.optics import matrix_K
from core.undistort import compute_housing_geometry, build_undistort_map_closed_form
from core.undistort_newton import build_undistort_map_newton


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_depth(name, depth_dir, z0_fixed, W, H):
    if depth_dir is None:
        return z0_fixed
    depth_path = f"{depth_dir}/{name}.npy"
    if not os.path.exists(depth_path):
        return None
    depth = np.load(depth_path).astype(np.float32)
    if depth.shape[:2] != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth, 1e-3, None)


def remap_with_mask(img, undist_x, undist_y):
    """Remap the image and compute a binary validity mask
    (255 = valid pixel, 0 = out-of-bounds / black region)."""
    h_src, w_src = img.shape[:2]

    valid = (
        (undist_x >= 0) & (undist_x <= w_src - 1) &
        (undist_y >= 0) & (undist_y <= h_src - 1)
    )

    corrected = cv2.remap(
        img, undist_x, undist_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    mask = np.where(valid, 255, 0).astype(np.uint8)
    return corrected, mask


def to_rgba(corrected_bgr, mask):
    """Combine a BGR image and a binary mask into an RGBA image
    (alpha channel = mask, for use with 3DGS/nerfstudio)."""
    bgra = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = mask
    return bgra


def main():
    parser = argparse.ArgumentParser(description="Refraction removal from a YAML config")
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

    paths = cfg["paths"]
    rgb_dir = paths["rgb_dir"]
    output_dir = paths["output_dir"]
    mask_dir = paths["mask_dir"]
    depth_dir = paths.get("depth_dir")

    corr = cfg["correction"]
    z0_fixed = corr["z0_fixed"]
    zoom = corr["zoom"]
    step_size = corr.get("step_size", 1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    K, K_inv = matrix_K(fx, fy, cx, cy)

    if method == "closed_form":
        P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w)
    else:
        kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                      mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)

    IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
                  "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]
    rgb_paths = sorted(
        p for ext in IMAGE_EXTS for p in glob.glob(f"{rgb_dir}/{ext}")
    )
    rgb_paths = rgb_paths[::step_size]

    print(f"Found {len(rgb_paths)} images (step={step_size}), method={method}")

    precomputed_map = None
    precomputed_mask = None
    if depth_dir is None:
        print(f"Single depth mode (Z0={z0_fixed}) — computing undistortion map once")
        if method == "closed_form":
            undist_x, undist_y = build_undistort_map_closed_form(
                P2, ray_water, z0_fixed, fx, fy, cx, cy, H, W, zoom=zoom
            )
        else:
            undist_x, undist_y = build_undistort_map_newton(
                fx, fy, cx, cy, z0_fixed, kwargs, W, H, zoom=zoom
            )
        undist_x = np.nan_to_num(undist_x, nan=-1)
        undist_y = np.nan_to_num(undist_y, nan=-1)
        precomputed_map = (undist_x, undist_y)
        precomputed_mask = np.where(
            (undist_x >= 0) & (undist_x <= W - 1) &
            (undist_y >= 0) & (undist_y <= H - 1),
            255, 0
        ).astype(np.uint8)
        cv2.imwrite(f"{mask_dir}/camera_mask.png", precomputed_mask)
        print(f"Saved shared mask: {mask_dir}/camera_mask.png")

    for rgb_path in rgb_paths:
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        out_path = f"{output_dir}/{name}.png"
        mask_path = f"{mask_dir}/{name}.png.png"
        if os.path.exists(out_path):
            continue

        img = cv2.imread(rgb_path)
        if img is None:
            continue

        if precomputed_map is not None:
            undist_x, undist_y = precomputed_map
            corrected = cv2.remap(
                img, undist_x, undist_y, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            mask = precomputed_mask
        else:
            depth = load_depth(name, depth_dir, z0_fixed, W, H)
            if depth is None:
                print(f"No depth: {name} → skip")
                continue

            if method == "closed_form":
                ux, uy = build_undistort_map_closed_form(
                    P2, ray_water, depth, fx, fy, cx, cy, H, W, zoom=zoom
                )
            else:
                ux, uy = build_undistort_map_newton(
                    fx, fy, cx, cy, depth, kwargs, W, H, zoom=zoom or 1.4
                )
            undist_x = np.nan_to_num(ux, nan=-1)
            undist_y = np.nan_to_num(uy, nan=-1)
            corrected, mask = remap_with_mask(img, undist_x, undist_y)
            cv2.imwrite(mask_path, mask)

        corrected_rgba = to_rgba(corrected, mask)
        cv2.imwrite(out_path, corrected_rgba)
        print(f"Saved: {name}.png ({corrected_rgba.shape[1]}x{corrected_rgba.shape[0]})")

    print("Done.")


if __name__ == "__main__":
    main()