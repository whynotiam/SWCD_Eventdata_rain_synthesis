import os
import argparse

import numpy as np
import pandas as pd
import pycolmap
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.interpolate import interp1d

# Project root auto-detection (overridable via WORKSPACE_DIR env var).
# This file lives in <repo>/colmap_workspace/, so go up one directory to reach the workspace root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.dirname(_THIS_DIR))


def extract_and_sync_poses(seq_name, colmap_dir):
    print(f"Loading COLMAP results for [{seq_name}]...")

    ts_file = os.path.join(WORKSPACE_DIR, "data", "DSEC", seq_name, "images", "left", "image_timestamps.txt")
    out_file = os.path.join(WORKSPACE_DIR, "data", "DSEC", seq_name, "ground_truth", "poses_colmap.csv")

    reconstruction = pycolmap.Reconstruction(colmap_dir)
    timestamps = pd.read_csv(ts_file, header=None).values.flatten()

    # Extract poses that survived COLMAP reconstruction (World-to-Camera -> Camera-to-World).
    # Compatible with both legacy (image.qvec) and modern (image.cam_from_world) pycolmap APIs.
    extracted_poses = {}
    for image_id, image in reconstruction.images.items():
        frame_idx = int(image.name.split('.')[0])

        if hasattr(image, 'qvec'):
            # Legacy pycolmap
            q = image.qvec
            t = image.tvec
        else:
            # Modern pycolmap: image.cam_from_world (attribute or method)
            pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
            rot = pose.rotation() if callable(pose.rotation) else pose.rotation
            q = rot.quat() if hasattr(rot, 'quat') and callable(rot.quat) else rot.quat
            if callable(q):
                q = q()
            t = pose.translation() if callable(pose.translation) else pose.translation

        # pycolmap quaternion order (qw, qx, qy, qz) -> scipy order (qx, qy, qz, qw)
        rot_mat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

        # Camera position in world coordinates: C = -R^T * t
        c_pos = -rot_mat.T @ t
        c_quat = R.from_matrix(rot_mat.T).as_quat()  # [qx, qy, qz, qw]
        extracted_poses[frame_idx] = {'pos': c_pos, 'quat': c_quat}

    print(f"COLMAP reconstructed {len(extracted_poses)} / {len(timestamps)} frames.")

    # Interpolate missing frames: linear for translation, SLERP for rotation
    valid_indices = sorted(list(extracted_poses.keys()))
    valid_times = timestamps[valid_indices]
    valid_pos = np.array([extracted_poses[i]['pos'] for i in valid_indices])
    valid_quats = np.array([extracted_poses[i]['quat'] for i in valid_indices])

    pos_interp = interp1d(valid_times, valid_pos, axis=0, kind='linear', fill_value="extrapolate")
    slerp = Slerp(valid_times, R.from_quat(valid_quats))

    final_poses = []
    for ts in timestamps:
        # SLERP does not extrapolate, so clip to the valid range
        ts_clipped = np.clip(ts, valid_times[0], valid_times[-1])
        pos = pos_interp(ts)
        quat = slerp(ts_clipped).as_quat()
        final_poses.append([ts, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])

    df = pd.DataFrame(final_poses, columns=['ts', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved interpolated poses to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, required=True)
    parser.add_argument('--colmap_dir', type=str, required=True)
    args = parser.parse_args()
    extract_and_sync_poses(args.seq_name, args.colmap_dir)