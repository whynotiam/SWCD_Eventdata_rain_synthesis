import numpy as np


class SpatialMiddleware:
    def __init__(self, config):
        # DSEC camRect1 (Left RGB Frame) intrinsics
        self.fx = config.cam['fx']
        self.fy = config.cam['fy']
        self.cx = config.cam['cx']
        self.cy = config.cam['cy']

        # Render resolution
        self.width = config.cam['width']
        self.height = config.cam['height']

    def project_and_cull(self, physics_state, sync_state):
        """
        physics_state: {'pos': (N,3), 'diam': (N,), 'vel': (N,3)} from PhysicsEngine
        sync_state:    {'depth': (H, W), ...} from DSECSyncManager
        """
        pos_3d = physics_state['pos']
        diameters = physics_state['diam']
        velocities = physics_state['vel']
        depth_map = sync_state['depth']

        # 1) First filter: discard particles behind the camera (Z <= 0)
        mask = pos_3d[:, 2] > 0.1
        pos_3d = pos_3d[mask]
        diameters = diameters[mask]
        velocities = velocities[mask]

        if len(pos_3d) == 0:
            return None

        Z = pos_3d[:, 2]

        # 2) Pinhole 3D -> 2D projection (vectorized, sub-pixel float32 preserved)
        u = (self.fx * pos_3d[:, 0] / Z) + self.cx
        v = (self.fy * pos_3d[:, 1] / Z) + self.cy

        # 2D pixel diameter with perspective scaling (mm -> meter via 0.001)
        diam_pix = (self.fx * (0.001 * diameters)) / Z

        # 3) Second filter: frustum culling (off-screen particles)
        valid_mask = (u > -10) & (u < self.width + 10) & (v > -10) & (v < self.height + 10)
        u = u[valid_mask]
        v = v[valid_mask]
        Z = Z[valid_mask]
        diam_pix = diam_pix[valid_mask]
        velocities = velocities[valid_mask]

        if len(u) == 0:
            return None

        # 4) Z-buffer occlusion against the dense depth map.
        # Round only for indexing; keep sub-pixel u/v for the renderer.
        u_idx = np.clip(np.round(u), 0, self.width - 1).astype(np.int32)
        v_idx = np.clip(np.round(v), 0, self.height - 1).astype(np.int32)

        # Background depth at (v, u)
        bg_depths = depth_map[v_idx, u_idx]

        # Handle holes where LiDAR has no measurement (sky / far void).
        # In DSEC these usually appear as 0, negative, NaN, or Inf.
        # Such regions are effectively at infinity, so raindrops must survive there.
        is_invalid_depth = (bg_depths <= 0) | np.isnan(bg_depths) | np.isinf(bg_depths)

        # Keep particles closer to the camera than the background, or where the background is unknown (sky).
        keep_mask = (Z < bg_depths) | is_invalid_depth

        # Return only the particles that will actually be drawn
        return {
            'u': u[keep_mask],                # float (sub-pixel x)
            'v': v[keep_mask],                # float (sub-pixel y)
            'z': Z[keep_mask],
            'diam_pix': diam_pix[keep_mask],  # float (pixel radius)
            'vel': velocities[keep_mask],     # instantaneous 3D velocity (for motion blur)
        }