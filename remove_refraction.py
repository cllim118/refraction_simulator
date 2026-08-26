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

    bgra = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = mask
    return bgra

def find_largest_valid_rectangle(mask):

    H, W = mask.shape
    valid = (mask > 0).astype(np.int32)

    height = np.zeros(W, dtype=np.int32)
    best_area = 0
    best_box = (0, H, 0, W)  # fallback: whole image

    for y in range(H):
        height = np.where(valid[y] > 0, height + 1, 0)

        stack = []  # stores (start_x, h)
        x = 0
        while x <= W:
            h = height[x] if x < W else 0
            start = x
            while stack and stack[-1][1] > h:
                s_x, s_h = stack.pop()
                width = x - s_x
                area = s_h * width
                if area > best_area:
                    best_area = area
                    best_box = (y - s_h + 1, y + 1, s_x, x)
                start = s_x
            stack.append((start, h))
            x += 1

    return best_box

def crop_to_inscribed_rectangle(img, box):

    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def adjust_intrinsics_for_crop(cx, cy, box):

    y0, y1, x0, x1 = box
    return cx - x0, cy - y0


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
    crop_valid_bbox = corr.get("crop_valid_bbox", False)

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

    print(f"Found {len(rgb_paths)} images (step={step_size}), method={method}, "
          f"crop_valid_bbox={crop_valid_bbox}")

    precomputed_map = None
    precomputed_mask = None
    precomputed_crop_box = None

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
        cv2.imwrite(f"{mask_dir}/mask.png", precomputed_mask)
        print(f"Saved shared mask: {mask_dir}/mask.png")

        if crop_valid_bbox:
            precomputed_crop_box = find_largest_valid_rectangle(precomputed_mask)
            y0, y1, x0, x1 = precomputed_crop_box
            new_cx, new_cy = adjust_intrinsics_for_crop(cx, cy, precomputed_crop_box)
            print(f"Inscribed valid rectangle: rows[{y0}:{y1}], cols[{x0}:{x1}] "
                  f"({x1 - x0}x{y1 - y0}, no invalid pixels)")
            print(f"Adjusted intrinsics for cropped output: "
                  f"fx={fx}, fy={fy}, cx={new_cx:.2f}, cy={new_cy:.2f}")

    for rgb_path in rgb_paths:
        name = os.path.splitext(os.path.basename(rgb_path))[0]
        out_path = f"{output_dir}/{name}.png"
        mask_path = f"{mask_dir}/{name}.png"
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
            crop_box = precomputed_crop_box
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

            crop_box = find_largest_valid_rectangle(mask) if crop_valid_bbox else None

        corrected_rgba = to_rgba(corrected, mask)

        if crop_valid_bbox and crop_box is not None:
            corrected_rgba = crop_to_inscribed_rectangle(corrected_rgba, crop_box)
            # No invalid pixels remain in this crop — drop alpha,
            # save as plain RGB (e.g. for VGGT-style pipelines that
            # can't consume a mask/alpha channel).
            corrected_out = cv2.cvtColor(corrected_rgba, cv2.COLOR_BGRA2BGR)
        else:
            # Keep RGBA with mask, as before (uncropped output).
            corrected_out = corrected_rgba

        cv2.imwrite(out_path, corrected_out)
        print(f"Saved: {name}.png ({corrected_out.shape[1]}x{corrected_out.shape[0]}, "
              f"{corrected_out.shape[2]} channels)")

    print("Done.")


if __name__ == "__main__":
    main()