import os
import sys
import argparse
from collections import deque

import cv2
import h5py
import numpy as np
import torch

# Project root auto-detection so the v2e import resolves no matter where the script is launched.
# This file lives in <repo>/src/, so go up one directory to reach the workspace root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.dirname(_THIS_DIR))
sys.path.append(os.path.join(WORKSPACE_DIR, "third_party", "v2e"))

from v2ecore.emulator import EventEmulator  # noqa: E402

from config import Config  # noqa: E402
from sync_manager import DSECSyncManager  # noqa: E402
from physics_engine import PhysicsEngine  # noqa: E402
from spatial_middleware import SpatialMiddleware  # noqa: E402
from renderer import RainRenderer  # noqa: E402

from utils.coordinate_utils import get_relative_transform  # noqa: E402
from utils.image_utils import warp_background_fast, save_integrated_rgb  # noqa: E402


class V2eWrapper:
    """Pipes rendered frame tensors directly into v2e and streams events to .h5."""

    def __init__(self, cfg, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.h5_path = os.path.join(output_dir, "events_synthetic.h5")

        self.emulator = EventEmulator(
            pos_thres=cfg.v2e_pos_thres,
            neg_thres=cfg.v2e_neg_thres,
            sigma_thres=cfg.v2e_sigma_thres,
            cutoff_hz=cfg.v2e_cutoff_hz,
            leak_rate_hz=cfg.v2e_leak_rate_hz,
            shot_noise_rate_hz=cfg.v2e_shot_noise_rate_hz,
        )

        self.h5_file = h5py.File(self.h5_path, 'w')
        self.event_dataset = self.h5_file.create_dataset(
            'events', shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=True
        )
        self.event_count = 0

    def process_frame(self, frame_rgb_np, timestamp_us):
        frame_gray = cv2.cvtColor(frame_rgb_np, cv2.COLOR_RGB2GRAY)
        events = self.emulator.generate_events(frame_gray, timestamp_us / 1e6)

        if events is not None and len(events) > 0:
            num_events = len(events)
            self.event_dataset.resize(self.event_count + num_events, axis=0)

            x = events[:, 1]
            y = events[:, 2]
            t_us = events[:, 0] * 1e6
            p = events[:, 3]

            formatted_events = np.column_stack((x, y, t_us, p))
            self.event_dataset[self.event_count:] = formatted_events
            self.event_count += num_events

    def close(self):
        self.h5_file.close()
        print(f"Events saved: {self.event_count} events written to {self.h5_path}")


class RainSimulatorPipeline:
    def __init__(self, cfg, start_ms=0):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing rain simulation pipeline (Device: {self.device})")

        self.sync_mgr = DSECSyncManager(cfg)
        self.physics = PhysicsEngine(cfg)
        self.spatial = SpatialMiddleware(cfg)
        self.renderer = RainRenderer(cfg)

        self.output_dir = os.path.join(cfg.output_dir, cfg.seq_name)
        self.rgb_out_dir = os.path.join(self.output_dir, "rgb_frames")
        os.makedirs(self.rgb_out_dir, exist_ok=True)

        self.v2e_streamer = V2eWrapper(cfg, self.output_dir)

        self.exposure_ms = cfg.exposure_time_ms
        self.rgb_buffer = deque(maxlen=self.exposure_ms)

        H, W = cfg.cam['height'], cfg.cam['width']
        self.static_noise_bg = torch.rand((1, 3, H, W), device=self.device)

    def run(self, start_ms=0, end_ms=None):
        if end_ms is None:
            end_ms = int((self.sync_mgr.end_ts - self.sync_mgr.start_ts) / 1000)

        print(f"Running simulation: {start_ms}ms ~ {end_ms}ms")

        try:
            for current_ms in range(start_ms, end_ms):
                sync_state = self.sync_mgr.get_state(current_ms)

                self.physics.update_particles(sync_state)
                drop_states = self.spatial.project_and_cull(self.physics.get_render_state(), sync_state)

                # Normalize background tensor (handle numpy / channel layouts / 0-255 vs 0-1)
                bg_data = sync_state['bg_rgb']
                if isinstance(bg_data, torch.Tensor):
                    bg_tensor = bg_data.float().to(self.device)
                else:
                    bg_tensor = torch.from_numpy(bg_data).float().to(self.device)

                if bg_tensor.max() > 2.0:
                    bg_tensor = bg_tensor / 255.0

                if bg_tensor.ndim == 3 and bg_tensor.shape[0] == 3:
                    pass
                elif bg_tensor.ndim == 3 and bg_tensor.shape[-1] == 3:
                    bg_tensor = bg_tensor.permute(2, 0, 1)

                # Warp the source RGB into the current 1ms camera viewpoint
                T_rel = get_relative_transform(
                    sync_state['R'], sync_state['t'],
                    sync_state['src_R'], sync_state['src_t'],
                    self.device,
                )

                warped_bg = warp_background_fast(
                    bg_tensor,
                    sync_state['depth'],
                    T_rel,
                    self.cfg.cam,
                    self.device,
                )

                # Stream A: render raindrops over the warped background (RGB output)
                crisp_rgb_frame, gt_mask_tensor = self.renderer.render(warped_bg.unsqueeze(0), drop_states)

                # Stream B: render raindrops over a static noise canvas (for v2e). Mask not needed here.
                crisp_event_frame, _ = self.renderer.render(self.static_noise_bg, drop_states)

                # Accumulate 1ms frames into the RGB buffer for motion-blur integration.
                # Offload to CPU to avoid GPU OOM.
                self.rgb_buffer.append(crisp_rgb_frame.detach().cpu())

                frame_np = (crisp_event_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                self.v2e_streamer.process_frame(frame_np, sync_state['ts'])

                if current_ms % 50 == 0:
                    # Save motion-blurred RGB by integrating the accumulated buffer
                    if len(self.rgb_buffer) > 0:
                        save_integrated_rgb(self.rgb_buffer, self.rgb_out_dir, current_ms)

                    # Save the pure GT mask (sharp snapshot, not integrated)
                    gt_mask_np = (gt_mask_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    mask_out_dir = os.path.join(self.output_dir, "gt_masks")
                    os.makedirs(mask_out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(mask_out_dir, f"mask_{current_ms:06d}.png"), gt_mask_np)

                if current_ms % 125 == 0:
                    print(f"Progress: {current_ms}ms processed")
        finally:
            self.v2e_streamer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_name', type=str, required=True, help="Sequence name to process")
    parser.add_argument('--start', type=int, default=0, help="Start time in ms")
    parser.add_argument('--end', type=int, default=None, help="End time in ms")
    args = parser.parse_args()

    cfg = Config(seq_name=args.seq_name)
    pipeline = RainSimulatorPipeline(cfg)

    total_duration_ms = int((pipeline.sync_mgr.end_ts - pipeline.sync_mgr.start_ts) / 1000)
    print(f"Detected sequence total length: {total_duration_ms} ms")

    actual_end = args.end if args.end is not None else total_duration_ms
    actual_end = min(actual_end, total_duration_ms)

    pipeline.run(start_ms=args.start, end_ms=actual_end)