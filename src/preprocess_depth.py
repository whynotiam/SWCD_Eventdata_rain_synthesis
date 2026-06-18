import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

# Project root auto-detection (overridable via WORKSPACE_DIR env var).
# This file lives in <repo>/src/, so go up one directory to reach the workspace root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.dirname(_THIS_DIR))


def process_disparity_to_depth(seq_name):
    print(f"[{seq_name}] Starting disparity-to-depth conversion...")

    base_dir = os.path.join(WORKSPACE_DIR, "data", "DSEC", seq_name)
    disp_dir = os.path.join(base_dir, "ground_truth", "disparity")
    out_dir = os.path.join(base_dir, "lidar", "event_cam_coords")
    calib_file = os.path.join(base_dir, "calibration", "cam_to_cam.yaml")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Parse DSEC official Q matrix dynamically (disparity_to_depth -> cams_12)
    with open(calib_file, 'r') as f:
        calib_data = yaml.safe_load(f)

    q_matrix = calib_data['disparity_to_depth']['cams_12']
    focal_length_x = float(q_matrix[2][3])
    inverse_baseline = float(q_matrix[3][2])
    focal_x_baseline = focal_length_x / inverse_baseline

    disp_files = sorted([f for f in os.listdir(disp_dir) if f.endswith('.png')])
    total_files = len(disp_files)

    if total_files == 0:
        print(f"Error: No .png files found in {disp_dir}")
        return

    valid_count = 0

    for idx, filename in enumerate(disp_files):
        img_path = os.path.join(disp_dir, filename)

        # Load 16-bit PNG disparity image
        disp_16bit = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if disp_16bit is None:
            continue

        # Convert to real disparity values per DSEC convention
        disp_float = disp_16bit.astype(np.float32) / 256.0

        depth_map = np.zeros_like(disp_float)

        # Avoid division by zero: compute depth only for valid disparities
        valid_mask = disp_float > 0.0

        # Z = (f * B) / d using the precomputed constant
        depth_map[valid_mask] = focal_x_baseline / disp_float[valid_mask]

        # Depth completion (hole filling): smoothly fill empty pixels using surrounding valid depths
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        depth_map_closed = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
        depth_map = np.where(depth_map == 0, depth_map_closed, depth_map)

        # Push sky / far regions (Depth=0) to "infinity" (1000m) to prevent false raindrop collisions
        depth_map = np.where(depth_map == 0, 1000.0, depth_map)

        out_filename = filename.replace('.png', '.npy')
        np.save(os.path.join(out_dir, out_filename), depth_map)
        valid_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            print(f"Progress: {idx + 1} / {total_files} converted")

    print(f"[{seq_name}] Conversion complete. {valid_count} depth maps saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, required=True, help="Sequence name (e.g., zurich_city_04_a)")
    args = parser.parse_args()
    process_disparity_to_depth(args.seq_name)