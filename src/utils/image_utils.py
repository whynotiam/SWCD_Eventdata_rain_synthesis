import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def warp_background_fast(bg_tensor, depth_map, T_rel, cam, device):
    """
    Fast inverse geometric warping for novel view synthesis.
    Uses the camera pose (T_rel) and a Z-buffer to interpolate a 1ms-displaced view
    without invoking a heavy optical-flow model (e.g. RAFT).
    """
    # Unpack camera intrinsics (passed in from config.py)
    W, H = cam['width'], cam['height']
    fx, fy, cx, cy = cam['fx'], cam['fy'], cam['cx'], cam['cy']

    # 1) Generate target-frame pixel coordinates (current 1ms view)
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij',
    )

    # 2) Back-project to the camera frame using normalized camera coordinates
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    # Convert to 3D point cloud (in current 1ms camera frame) using the Z-buffer
    Z = torch.tensor(depth_map, device=device, dtype=torch.float32).view(-1)
    valid_mask = Z > 0.0

    X = x_norm.view(-1) * Z
    Y = y_norm.view(-1) * Z

    P_target = torch.stack([X, Y, Z, torch.ones_like(Z)], dim=0)  # (4, H*W)

    # 3) Transform to the source camera frame
    P_source = T_rel @ P_target
    X_src, Y_src, Z_src = P_source[0], P_source[1], P_source[2]

    # Guard against pixels behind the source camera
    Z_src = torch.clamp(Z_src, min=1e-6)

    # 4) Reproject onto the source image plane
    u_src = (fx * X_src / Z_src) + cx
    v_src = (fy * Y_src / Z_src) + cy

    # Scale to [-1, 1] for grid_sample
    u_norm = (u_src / (W - 1)) * 2 - 1
    v_norm = (v_src / (H - 1)) * 2 - 1

    warp_grid = torch.stack((u_norm, v_norm), dim=-1).view(1, H, W, 2)

    # 5) Single warp pass for the whole image
    warped_bg = F.grid_sample(
        bg_tensor.unsqueeze(0), warp_grid, mode='bilinear', padding_mode='border', align_corners=False
    )

    # Fall back to the original pixel where depth is invalid (sky / Z=0) to prevent tearing
    bg_unsq = bg_tensor.unsqueeze(0)
    valid_mask_2d = valid_mask.view(1, 1, H, W)
    warped_bg = torch.where(valid_mask_2d, warped_bg, bg_unsq)

    return warped_bg.squeeze(0)


def save_integrated_rgb(rgb_buffer, out_dir, current_ms):
    """
    Linearly integrate the 20ms circular buffer to produce a motion-blurred frame and save it.
    """
    integrated_rgb = torch.mean(torch.stack(list(rgb_buffer)), dim=0)
    out_img = (integrated_rgb.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_path = os.path.join(out_dir, f"frame_{current_ms:06d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))