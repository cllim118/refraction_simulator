import os
import sys
import glob
import argparse
import yaml
import numpy as np
import cv2

from core.optics import matrix_K, apply_radtan_distortion
from core.undistort import compute_housing_geometry, build_undistort_map_closed_form
from core.undistort_newton import build_undistort_map_newton
from core.find_best_scale import find_optimal_scale, find_in_bounds_scale

IMAGE_EXTS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG",
              "*.bmp", "*.BMP", "*.tiff", "*.TIFF"]


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


def compute_valid_mask(undist_x, undist_y, w_src, h_src):
    """255 = source pixel is in bounds, 0 = out-of-bounds / black region."""
    valid = (
        (undist_x >= 0) & (undist_x <= w_src - 1) &
        (undist_y >= 0) & (undist_y <= h_src - 1)
    )
    return np.where(valid, 255, 0).astype(np.uint8)


def remap_with_mask(img, undist_x, undist_y):
    """Remap the image and compute its validity mask."""
    h_src, w_src = img.shape[:2]
    corrected = cv2.remap(
        img, undist_x, undist_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    mask = compute_valid_mask(undist_x, undist_y, w_src, h_src)
    return corrected, mask


def to_rgba(corrected_bgr, mask):
    bgra = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = mask
    return bgra


def find_largest_valid_rectangle(mask):
    H, W = mask.shape
    valid = mask > 0

    height = np.zeros(W, dtype=np.int32)
    best_area = 0
    best_box = (0, H, 0, W)  # fallback: whole image

    for y in range(H):
        height = np.where(valid[y], height + 1, 0)

        stack = []  # (start_x, h)
        for x in range(W + 1):
            h = height[x] if x < W else 0
            start = x
            while stack and stack[-1][1] > h:
                s_x, s_h = stack.pop()
                area = s_h * (x - s_x)
                if area > best_area:
                    best_area = area
                    best_box = (y - s_h + 1, y + 1, s_x, x)
                start = s_x
            stack.append((start, h))

    return best_box


def crop_to_inscribed_rectangle(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def adjust_intrinsics_for_crop(cx, cy, box):
    y0, y1, x0, x1 = box
    return cx - x0, cy - y0


def write_output_calibration(path, W, H, fx, fy, cx, cy, s):
    with open(path, "w") as f:
        f.write(f"focal_length: [{fx:.6f}, {fy:.6f}]\n")
        f.write(f"principal_point: [{cx:.6f}, {cy:.6f}]\n")
        f.write("distortion_coefficients: [0, 0, 0, 0]\n")
        f.write(f"image_dimension: [{W}, {H}]\n")
        f.write(f"zoom: {s:.6f}\n")

    print(f"Saved calibration: {path}")


def find_rgb_paths(rgb_dir, step_size):
    paths = sorted(p for ext in IMAGE_EXTS for p in glob.glob(f"{rgb_dir}/{ext}"))
    return paths[::step_size]


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

    k1, k2 = cam.get("k1", 0.0), cam.get("k2", 0.0)
    p1, p2 = cam.get("p1", 0.0), cam.get("p2", 0.0)
    apply_radtan = any(v != 0.0 for v in (k1, k2, p1, p2))

    hs = cfg["housing"]
    n_port = np.array(hs["n_port"])
    mu_a, mu_g, mu_w = hs["mu_a"], hs["mu_g"], hs["mu_w"]
    rflat, tglass = hs["rflat"], hs["tglass"]

    paths = cfg["paths"]
    rgb_dir = paths["rgb_dir"]
    output_dir = paths["output_dir"]
    mask_dir = paths["mask_dir"]
    depth_dir = paths.get("depth_dir")
    calib_path = paths.get("calib_path",os.path.join(os.path.dirname(output_dir), "new_calibration.yaml"))

    corr = cfg["correction"]
    z0_fixed = corr["z0_fixed"]
    step_size = corr.get("step_size", 1)
    crop_valid_bbox = corr.get("crop_valid_bbox", False)

    print("Finding optimal zoom...")

    map_x, map_y, s_values, rmse_values, _, _ = find_optimal_scale(
        z0_fixed, W, H, fx, fy, cx, cy,
        n_port, rflat, tglass, mu_a, mu_g, mu_w
    )

    zoom, zoom_rmse = find_in_bounds_scale(
        map_x, map_y, s_values, rmse_values,
        W, H, cx, cy
    )

    if zoom is None:
        raise RuntimeError(
            "No in-bounds zoom found in the search range."
        )

    print(f"Best scale in bounds = {zoom:.4f}")
    print(f"RMSE = {zoom_rmse:.4f} px")


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    K, K_inv = matrix_K(fx, fy, cx, cy)

    if method == "closed_form":
        P2, ray_water = compute_housing_geometry(H, W, K_inv, n_port, rflat, tglass, mu_a, mu_g, mu_w)
    else:
        newton_kwargs = dict(K_inv=K_inv, n_port=n_port, rflat=rflat, tglass=tglass,
                              mu_a=mu_a, mu_g=mu_g, mu_w=mu_w)

    def build_map(z_or_depth, zoom_val):
        if method == "closed_form":
            return build_undistort_map_closed_form(
                P2, ray_water, z_or_depth, fx, fy, cx, cy, H, W, zoom=zoom_val
            )
        return build_undistort_map_newton(
            fx, fy, cx, cy, z_or_depth, newton_kwargs, W, H, zoom=zoom_val
        )

    def apply_radtan_to_map(map_x, map_y):
        x_norm = (map_x - cx) / fx
        y_norm = (map_y - cy) / fy
        x_dist, y_dist = apply_radtan_distortion(x_norm, y_norm, k1, k2, p1, p2)
        return x_dist * fx + cx, y_dist * fy + cy

    def finalize_map(ux, uy):
        if apply_radtan:
            ux, uy = apply_radtan_to_map(ux, uy)
        return np.nan_to_num(ux, nan=-1), np.nan_to_num(uy, nan=-1)

    rgb_paths = find_rgb_paths(rgb_dir, step_size)
    print(f"Found {len(rgb_paths)} images (step={step_size}), method={method}, "
          f"crop_valid_bbox={crop_valid_bbox}, radtan={apply_radtan}")

    precomputed_map = None
    precomputed_mask = None
    precomputed_crop_box = None

    if depth_dir is None:
        print(f"Single depth mode (Z0={z0_fixed}) — computing undistortion map once")
        zoom_eff = zoom if method == "closed_form" else (zoom or 1.4)
        undist_x, undist_y = finalize_map(*build_map(z0_fixed, zoom))
        precomputed_map = (undist_x, undist_y)
        precomputed_mask = compute_valid_mask(undist_x, undist_y, W, H)
        cv2.imwrite(f"{mask_dir}/mask.png", precomputed_mask)
        print(f"Saved shared mask: {mask_dir}/mask.png")

        out_W, out_H = W, H
        out_fx, out_fy = fx * zoom_eff, fy * zoom_eff
        out_cx, out_cy = cx, cy

        if crop_valid_bbox:
            precomputed_crop_box = find_largest_valid_rectangle(precomputed_mask)
            y0, y1, x0, x1 = precomputed_crop_box
            new_cx, new_cy = adjust_intrinsics_for_crop(out_cx, out_cy, precomputed_crop_box)
            out_W, out_H = x1 - x0, y1 - y0
            out_cx, out_cy = new_cx, new_cy
            print(f"Inscribed valid rectangle: rows[{y0}:{y1}], cols[{x0}:{x1}] "
                  f"({x1 - x0}x{y1 - y0}, no invalid pixels)")
            print(f"Adjusted intrinsics for cropped output: "
                  f"fx={out_fx:.2f}, fy={out_fy:.2f}, cx={new_cx:.2f}, cy={new_cy:.2f}")

        write_output_calibration(calib_path, out_W, out_H,
                                 out_fx, out_fy, out_cx, out_cy, zoom)

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

            zoom_val = zoom if method == "closed_form" else (zoom or 1.4)
            undist_x, undist_y = finalize_map(*build_map(depth, zoom_val))
            corrected, mask = remap_with_mask(img, undist_x, undist_y)
            cv2.imwrite(mask_path, mask)

            crop_box = find_largest_valid_rectangle(mask) if crop_valid_bbox else None

        if crop_valid_bbox and crop_box is not None:
            # Crop first: every pixel inside the inscribed rectangle is valid,
            # so the alpha channel adds nothing and the BGRA round-trip is skipped.
            corrected_out = crop_to_inscribed_rectangle(corrected, crop_box)
        else:
            corrected_out = to_rgba(corrected, mask)

        cv2.imwrite(out_path, corrected_out)
        print(f"Saved: {name}.png ({corrected_out.shape[1]}x{corrected_out.shape[0]}, "
              f"{corrected_out.shape[2]} channels)")

    print("Done.")


if __name__ == "__main__":
    main()
