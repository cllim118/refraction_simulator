import numpy as np
import cv2
from pathlib import Path

# =========================================================
# Config
# =========================================================
INPUT_DIR  = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/mar/rgb_0"
OUTPUT_DIR = "/home/chelim/chelim/VSLAM-LAB-Benchmark/LIZARDISLAND/mar_corr/rgb_0"
MAP_PATH   = "undist_lizard_Z1.0_s1.40_1.yaml"
EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

# =========================================================
# Run
# =========================================================
if __name__ == "__main__":

    # Load undistortion map
    fs = cv2.FileStorage(MAP_PATH, cv2.FILE_STORAGE_READ)
    map_combined = fs.getNode("Undistortion Map").mat()
    best_s       = fs.getNode("best_s").real()
    fs.release()

    undist_x = map_combined[:, :, 0]
    undist_y = map_combined[:, :, 1]
    print(f"Loaded map: {MAP_PATH}, best_s={best_s:.4f}")

    # output folder
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # image paths
    input_dir = Path(INPUT_DIR)
    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS]
    print(f"Found {len(img_paths)} images")

    for img_path in sorted(img_paths):

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skipped (load failed): {img_path.name}")
            continue

        img_corrected = cv2.remap(img, undist_x, undist_y, cv2.INTER_LINEAR)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), img_corrected)
        print(f"Saved: {out_path}")

    print("Done.")