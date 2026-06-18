import os
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Slerp, Rotation as R
from scipy.interpolate import interp1d


class DSECSyncManager:
    def __init__(self, config):
        self.cfg = config
        self.base_path = config.base_path

        # Load camera poses (poses_colmap.csv produced by COLMAP + sync_poses.py)
        pose_df = pd.read_csv(os.path.join(self.base_path, "ground_truth", "poses_colmap.csv"))
        self.pose_ts = pose_df['ts'].values.astype(np.float64)
        self.positions = pose_df[['x', 'y', 'z']].values
        self.quaternions = pose_df[['qx', 'qy', 'qz', 'qw']].values

        # Fix the start timestamp (this becomes the 0ms reference)
        self.start_ts = self.pose_ts[0]
        self.end_ts = self.pose_ts[-1]

        # Rotation interpolation engine (SLERP)
        rotations = R.from_quat(self.quaternions)
        self.slerp = Slerp(self.pose_ts, rotations)

        # Position interpolation engine (linear)
        self.pos_interp = interp1d(self.pose_ts, self.positions, axis=0, kind='linear', fill_value="extrapolate")

        # Depth file paths and timestamps
        self.depth_dir = os.path.join(self.base_path, "lidar", "event_cam_coords")
        self.depth_ts = pd.read_csv(
            os.path.join(self.base_path, "ground_truth", "disparity_timestamps.txt"), header=None
        ).values.flatten()
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith('.npy')])

        # Original 20Hz RGB paths and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rgb_dir = os.path.join(self.base_path, "images", "left", "rectified")
        self.rgb_ts = pd.read_csv(
            os.path.join(self.base_path, "images", "left", "image_timestamps.txt"), header=None
        ).values.flatten()
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.png')])

    # RGB frame loader with cache to avoid I/O bottlenecks
    @lru_cache(maxsize=2)
    def _load_rgb_frame(self, idx):
        img = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float().to(self.device) / 255.0

    # Time flows forward only, so maxsize=2 is enough to eliminate I/O bottlenecks
    @lru_cache(maxsize=2)
    def _load_depth_map(self, depth_idx):
        return np.load(os.path.join(self.depth_dir, self.depth_files[depth_idx]))

    def get_state(self, current_ms):
        """
        Input: elapsed milliseconds since simulation start.
        Output: precise pose (R, t) and the nearest depth map at that instant.
        """
        # Absolute timestamp at 1ms resolution
        target_ts = self.start_ts + (current_ms * 1000)

        # Strict time clipping to keep SLERP within its valid range
        clipped_ts = np.clip(target_ts, self.start_ts, self.end_ts)

        # Pose: SLERP and linear interpolation densely fill 1ms steps inside the 50ms intervals
        curr_pos = self.pos_interp(clipped_ts)
        curr_rot = self.slerp(clipped_ts).as_matrix()  # 3x3 rotation matrix

        # Depth: nearest-neighbor lookup
        depth_idx = np.argmin(np.abs(self.depth_ts - target_ts))
        depth_idx = min(depth_idx, len(self.depth_files) - 1)  # guard against out-of-range
        depth_map = self._load_depth_map(depth_idx)

        # Nearest source RGB frame and its pose (used as the warping reference)
        rgb_idx = np.argmin(np.abs(self.rgb_ts - target_ts))
        rgb_idx = min(rgb_idx, len(self.rgb_files) - 1)
        bg_rgb = self._load_rgb_frame(rgb_idx)
        src_ts = self.rgb_ts[rgb_idx]

        # Pose at the moment the source RGB was captured
        src_pos = self.pos_interp(src_ts)
        src_rot = self.slerp(src_ts).as_matrix()

        return {
            'R': curr_rot,
            't': curr_pos,
            'depth': depth_map,
            'bg_rgb': bg_rgb,    # nearest source RGB tensor
            'src_R': src_rot,    # camera rotation when the source RGB was captured
            'src_t': src_pos,    # camera position when the source RGB was captured
            'ts': target_ts,
        }