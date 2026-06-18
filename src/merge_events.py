import os
import argparse

import h5py
import hdf5plugin  # noqa: F401  (required for DSEC's compressed events.h5)
import numpy as np


def merge_dsec_events(original_h5_path, synthetic_h5_path, output_h5_path):
    print("[Phase 4] Merging events to build InputEvent...")

    # 1) Load original DSEC events 
    print(f"Loading OEvent: {original_h5_path}")
    with h5py.File(original_h5_path, 'r') as f_orig:
        o_x = f_orig['events/x'][:]
        o_y = f_orig['events/y'][:]
        o_t = f_orig['events/t'][:]
        o_p = f_orig['events/p'][:]
        t_offset = f_orig['t_offset'][()]

    # DSEC original event time `t` is relative (starts from 0); add t_offset to get
    # absolute time (microseconds) so both event streams share the same clock.
    o_t_abs = o_t + t_offset
    print(f"   -> OEvent count: {len(o_x):,}")

    # 2) Load synthesized rain events
    print(f"Loading RDE: {synthetic_h5_path}")
    with h5py.File(synthetic_h5_path, 'r') as f_syn:
        syn_events = f_syn['events'][:]

    r_x = syn_events[:, 0].astype(np.uint16)
    r_y = syn_events[:, 1].astype(np.uint16)

    # main_parallel.py already passes absolute timestamps (microseconds) to v2e,
    # so RDE times are already on the same absolute clock as o_t_abs.
    r_t_abs = syn_events[:, 2].astype(np.int64)
    r_p = syn_events[:, 3].astype(np.int8)

    # v2e sometimes emits polarity as -1/1. Normalize to DSEC standard (0/1).
    r_p[r_p == -1] = 0
    print(f"   -> RDE count: {len(r_x):,}")

    # 3) Concatenate
    print("Merging and sorting by time...")
    merged_x = np.concatenate((o_x, r_x))
    merged_y = np.concatenate((o_y, r_y))
    merged_t_abs = np.concatenate((o_t_abs, r_t_abs))
    merged_p = np.concatenate((o_p, r_p))

    # Inherit the original DSEC t_offset as the new global time origin
    new_t_offset = t_offset

    # Drop any synthetic outliers that ended up before the original t_offset
    valid_mask = merged_t_abs >= new_t_offset
    merged_x = merged_x[valid_mask]
    merged_y = merged_y[valid_mask]
    merged_t_abs = merged_t_abs[valid_mask]
    merged_p = merged_p[valid_mask]

    # 4) Sort by absolute time
    sort_idx = np.argsort(merged_t_abs)
    merged_x = merged_x[sort_idx]
    merged_y = merged_y[sort_idx]
    merged_t_abs = merged_t_abs[sort_idx]
    merged_p = merged_p[sort_idx]

    # 5) Recompute relative time using the fixed t_offset
    merged_t_rel = merged_t_abs - new_t_offset

    # 6) Recompute the ms_to_idx index
    print("Computing ms_to_idx index...")
    max_ms = int(merged_t_rel[-1] // 1000)
    ms_values = np.arange(max_ms + 1) * 1000
    ms_to_idx = np.searchsorted(merged_t_rel, ms_values).astype(np.uint64)

    # 7) Write final InputEvent in DSEC official format
    print(f"Writing InputEvent: {output_h5_path}")
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    with h5py.File(output_h5_path, 'w') as f_out:
        events_grp = f_out.create_group('events')

        # h5py default lzf compression: good balance of read/write throughput and file size
        events_grp.create_dataset('x', data=merged_x, dtype=np.uint16, compression="lzf")
        events_grp.create_dataset('y', data=merged_y, dtype=np.uint16, compression="lzf")
        events_grp.create_dataset('t', data=merged_t_rel, dtype=np.int64, compression="lzf")
        events_grp.create_dataset('p', data=merged_p, dtype=np.int8, compression="lzf")

        f_out.create_dataset('t_offset', data=new_t_offset, dtype=np.int64)
        f_out.create_dataset('ms_to_idx', data=ms_to_idx, dtype=np.uint64)

    print("InputEvent generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=str, required=True, help="SLURM Job ID (or any run identifier)")
    parser.add_argument('--seq_name', type=str, required=True, help="Sequence name")
    args = parser.parse_args()

    from config import Config
    cfg = Config(seq_name=args.seq_name)

    # Output data folder layout: <output_dir>/<seq_name>/...
    job_dir = os.path.join(cfg.output_dir, cfg.seq_name)

    # 1) Input file paths
    o_event_path = os.path.join(cfg.base_path, "events", "left", "events.h5")
    rde_path = os.path.join(job_dir, "events_synthetic.h5")

    # 2) Output file path
    final_output_dir = os.path.join(job_dir, "final_input")
    os.makedirs(final_output_dir, exist_ok=True)
    input_event_path = os.path.join(final_output_dir, "InputEvent.h5")

    merge_dsec_events(o_event_path, rde_path, input_event_path)