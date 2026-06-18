# SWCD_Eventdata_rain_synthesis

# Physics-Based 3D Event-RGB Rain Synthesis for DSEC

---

## Overview

This project provides a 3D rain synthesis pipeline based on physical laws (Marshall–Palmer, Gunn–Kinzer) for DSEC, an Event–RGB dataset for autonomous driving. Rather than a simple 2D rain streak overlay, it generates synthetic data that simultaneously guarantees the following:

- **3D rain particle simulation in real space** — Generation grounded in actual depth information, leveraging the camera pose (R, t) and a Z-buffer
- **Spatio-temporal sync at 1ms (1000Hz) precision** — 20Hz raw RGB is interpolated via inverse geometric warping
- **Physical consistency of Event–RGB pairs** — The same physical computation is passed through a dualized rendering stream, producing both the RGB frame and the event mask GT at once
- **3D Occlusion handling** — LiDAR-based depth maps naturally hide raindrops behind buildings and vehicles

The generated data can be used for training event-based **Deraining** models. Benchmarking with a pretrained Restormer achieved a PSNR of **33.67dB** and NIQE of **5.1994**.

---

## Key Features

| Feature | Description |
|---|---|
| **Physical particle generation** | Raindrop diameters sampled from the Marshall–Palmer distribution; terminal velocity assigned via the Gunn–Kinzer model |
| **Spatio-temporal sync (1ms)** | Camera poses reconstructed at 1ms resolution via SLERP + linear interpolation; RGB interpolated via inverse geometric warping |
| **Z-buffer occlusion** | DSEC LiDAR disparity converted to depth, followed by per-pixel occlusion handling |
| **Dual-stream rendering** | Simultaneous output of an RGB stream (warped background + motion blur) and an Event stream (static noise canvas + v2e) |
| **Optical realism** | Sphere refraction, edge attenuation, optical bloom, atmospheric veiling, sensor noise |
| **Treadmill recycling** | After applying ego-motion, particles that leave the frame are re-placed to ensure long-term stability |

---

## Pipeline Architecture

The pipeline is organized into five sequential phases.

**Phase 0** — **Preprocessing**. Raw DSEC data is split into two preprocessing branches. 16-bit disparity PNG files are converted into metric `.npy` depth maps through `preprocess_depth.py`. In parallel, the left rectified images are passed through COLMAP for sparse reconstruction, and `sync_poses.py` extracts and interpolates the resulting poses into a `poses_colmap.csv` file aligned with every RGB timestamp.

**Phase 1** — **Spatio-Temporal Sync**. The `DSECSyncManager` (`sync_manager.py`) consolidates this preprocessed information. It performs SLERP-based rotation interpolation and linear position interpolation at 1ms resolution, while serving the nearest-neighbor depth maps and RGB frames through an LRU cache to avoid I/O bottlenecks.

**Phase 2** — **Physics & Spatial Middleware**. The `PhysicsEngine` (`physics_engine.py`) drives the particle dynamics. Raindrop diameters are drawn from the Marshall–Palmer distribution `N(D) = N₀ exp(-ΛD)` with `Λ = 4.1 R⁻⁰·²¹`, and each particle is assigned a Gunn–Kinzer terminal velocity `Vt(D) = 9.65 − 10.3 exp(−0.6D) [m/s]`. Vehicle ego-motion is back-projected onto the particles to keep them in the camera coordinate frame. The `SpatialMiddleware` (`spatial_middleware.py`) then projects the 3D particles onto the 2D image plane via a pinhole model (preserving sub-pixel float32 precision), and applies frustum culling along with Z-buffer occlusion.

**Phase 3** — **Dual-Stream Rendering**. The `RainRenderer` (`renderer.py`) splits the output into two streams that share the same physical computation. Stream A (RGB) composes the warped background with refraction, streaks, bloom, fog, and noise, then integrates 1ms frames over the 20ms exposure window to produce motion-blurred output saved as `rgb_frames/frame_XXXXXX.png`. Stream B (Event GT) executes the same rain computation on top of a static noise canvas and streams the result directly into the v2e EventEmulator, writing `events_synthetic.h5`.

**Phase 4** — **Event Merging**. Finally, `merge_events.py` merges the original DSEC `events.h5` with the synthetic rain events. Records are sorted by absolute timestamp, `ms_to_idx` is recomputed, and the result is exported as `InputEvent.h5` in the official DSEC format.

---

## Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (verified on RTX 3090 24GB)
- **RAM**: 32GB or higher recommended
- **Disk**: Approximately 50–100GB of output space per sequence

### Software
- Python ≥ 3.8
- CUDA ≥ 11.3
- COLMAP ≥ 3.7 (CLI usage)

### Python Dependencies
```bash
pip install numpy pandas scipy
pip install torch torchvision      # CUDA build
pip install opencv-python
pip install pycolmap
pip install h5py hdf5plugin
pip install PyYAML
```

### Third-party
- **[v2e](https://github.com/SensorsINI/v2e)** — Clone into `third_party/v2e/` 
  ```bash
  mkdir -p third_party && cd third_party
  git clone https://github.com/SensorsINI/v2e.git
  ```

---

## 📂 Repository Structure

```
Rain_Generator/
├── config.py                      # Dynamic paths / camera, physics, rendering parameters
├── sync_manager.py                # 1ms-resolution pose/Depth/RGB synchronization manager
├── physics_engine.py              # Marshall–Palmer + Gunn–Kinzer physics engine
├── spatial_middleware.py          # 3D→2D projection / Frustum culling / Z-buffer
├── renderer.py                    # Refraction, motion-blur, bloom, noise dual renderer
├── main_parallel.py               # Full pipeline orchestrator (entrypoint)
│
├── sync_poses.py                  # COLMAP result → poses_colmap.csv extraction & interpolation
├── preprocess_depth.py            # disparity(.png) → depth(.npy) conversion
├── merge_events.py                # Merge original + synthetic events → InputEvent.h5
│
├── utils/
│   └── coordinate_utils.py        # Relative transformation matrix T_rel computation
│
├── third_party/
│   └── v2e/                       # SensorsINI/v2e (clone required)
│
├── colmap_workspace/              # COLMAP sparse reconstruction results
│   └── colmap_results_<JOB_ID>/sparse/0/
│
└── data/
    ├── DSEC/<seq_name>/           # Input data
    │   ├── images/left/rectified/*.png
    │   ├── images/left/image_timestamps.txt
    │   ├── events/left/events.h5
    │   ├── ground_truth/disparity/*.png
    │   ├── ground_truth/disparity_timestamps.txt
    │   ├── ground_truth/poses_colmap.csv     # ← Output of sync_poses.py
    │   ├── lidar/event_cam_coords/*.npy      # ← Output of preprocess_depth.py
    │   └── calibration/cam_to_cam.yaml
    │
    └── synthetic_output/<seq_name>/<seq_name>/   # Output data
        ├── rgb_frames/frame_XXXXXX.png      # Motion-blurred RGB (50ms intervals)
        ├── gt_masks/mask_XXXXXX.png         # Pure rain GT mask
        ├── events_synthetic.h5              # RDE (Rain Drop Events)
        └── final_input/InputEvent.h5        # Merged result (original + synthetic)
```

---

## 🚀 Usage

### 0. Environment Variable Setup (recommended)
All scripts recognize `WORKSPACE_DIR`.
```bash
export WORKSPACE_DIR="/path/to/Rain_Generator"
```

### 1. Preparing DSEC Data

Download the following data frome
[DSEC official page](https://dsec.ifi.uzh.ch/) and place it under `data/DSEC/<seq_name>/` (example: `zurich_city_04_a`)

| Item | Path |
|---|---|
| Rectified left images | `images/left/rectified/*.png` |
| Image timestamps | `images/left/image_timestamps.txt` |
| Events | `events/left/events.h5` |
| Disparity | `ground_truth/disparity/*.png` |
| Disparity timestamps | `ground_truth/disparity_timestamps.txt` |
| Calibration | `calibration/cam_to_cam.yaml` |

### 2. Depth Map Preprocessing (Disparity → Depth)

`fx · baseline` is automatically parsed from the `disparity_to_depth.cams_12` matrix in `cam_to_cam.yaml`, and 16-bit disparity PNGs are converted to depth `.npy` files in meter units. Hole filling via morphological closing and sky-region (Depth=0) → 1000m handling are included.

```bash
python preprocess_depth.py --seq_name zurich_city_04_a
```
output: `data/DSEC/<seq_name>/lidar/event_cam_coords/*.npy`

### 3. COLMAP-based Camera Pose Extraction

DSEC provides IMU/GNSS but not precise camera poses, so **COLMAP** is used to perform sparse reconstruction.

```bash
# (Example) COLMAP CLI automation
colmap automatic_reconstructor \
    --workspace_path  ./colmap_workspace/colmap_results_<JOB_ID> \
    --image_path      ./data/DSEC/<seq_name>/images/left/rectified \
    --single_camera   1
```

Then, poses are extracted from the reconstruction result (sparse/0), and missing frames are interpolated via SLERP + linear interpolation to produce poses for every RGB timestamp.

```bash
python sync_poses.py \
    --seq_name zurich_city_04_a \
    --colmap_dir ./colmap_workspace/colmap_results_<JOB_ID>/sparse/0
```
Output: `data/DSEC/<seq_name>/ground_truth/poses_colmap.csv`
(Columns: `ts, x, y, z, qx, qy, qz, qw`)

> **Compatibility note**: `sync_poses.py` automatically detects and handles both `image.qvec`(legacy) and `image.cam_from_world`(newer) pycolmap API

### 4. Running the Main Pipeline (RGB + Event synthesis)

```bash
python main_parallel.py \
    --seq_name zurich_city_04_a \
    --start 0      \
    --end   7000      # 7 second window (in ms; omit for the full sequence)
```

Outputs:
- `data/synthetic_output/<seq_name>/<seq_name>/rgb_frames/frame_XXXXXX.png` — Motion-blurred synthetic RGB (50ms intervals)
- `data/synthetic_output/<seq_name>/<seq_name>/gt_masks/mask_XXXXXX.png` — Pixel-level rain GT mask
- `data/synthetic_output/<seq_name>/<seq_name>/events_synthetic.h5` — Rain drop events

### 5. Merging Original DSEC Events with Synthetic Events

Evets are merged in the DSEC standard format (`x`, `y`, `t`, `p`, `t_offset`, `ms_to_idx`). The original `t_offset` is herited, so the timeline is exactly aligned.

```bash
python merge_events.py \
    --job_id  <JOB_ID> \
    --seq_name zurich_city_04_a
```
Output: `data/synthetic_output/<seq_name>/<seq_name>/final_input/InputEvent.h5`

---

## ⚙️ Configuration

All hyperparameters are managed in one place in [`config.py`](config.py). The camera matrix is loaded dynamically from `cam_to_cam.yaml`

### 주요 파라미터

| 카테고리 | 파라미터 | 기본값 | 설명 |
|---|---|---|---|
| **Physics** | `rain_rate` | `10.0` mm/h | Rainfall rate (R) |
| | `box_width / height / depth` | `50 / 30 / 30` m | Rain generation 3D Bounding Box |
| **Render** | `exposure_time_ms` | `20` | Motion-blur integration exposure time |
| | `refraction_strength` | `1.8` | Raindrop refraction strength (S_refr) |
| | `specular_intensity` | `0.2` | Raindrop surface specular highlight |
| **v2e** | `v2e_pos_thres / neg_thres` | `0.2` | Event firing thresholds |
| | `v2e_sigma_thres` | `0.0` | For pure mask (noise blocked) |
| | `v2e_cutoff_hz / leak_rate / shot_noise` | `0` | All noise-related options disabled |

### Visual Effects Inside `renderer.py`

| 효과 | 변수 | 기본값 |
|---|---|---|
| Rain opacity | `rain_opacity` | `0.5` |
| Rain streak stretch (Y-axis) | `stretch_y` | `1000.0` |
| Maximum raindrop thickness | `clamp(d, max=8.0)` | `8 px` |
| Bloom threshold | `bloom_threshold` | `0.75` |
| Bloom intensity | `bloom_intensity` | `0.6` |
| Illumination attenuation (bad weather) | `illumination_factor` | `0.88` |
| Atmospheric veil denstiy | `fog_density` | `0.10` |
| Sensor noise σ | `noise_std_dev` | `0.02` |

---

## 📊 Results

Evaluation was performed on RGB–GT pairs of 7-second frames at 50ms intervals from DSEC daytime/nighttime sequences (GPU: RTX 3090 ×1).

### Physical Property Validation
- The pixel length of a rain streak is proportional to terminal velocity and inversely proportional to the camera distance → **fan-shaped distribution confirmed** 
- Independence between Marshall–Palmer diameter sampling and depth was verified

### Event–RGB Synchronization Precision
- Ratio of events within a 2ms window that fall inside the rain mask region: **0.6337**

### Restormer Benchmarking (Deraining)

**PSNR (dB)**

| | Test100 | Rain100H | Rain100L | Test2800 | Test200 | Average | **Ours** |
|---|---|---|---|---|---|---|---|
| Restormer | 32.00 | 31.46 | 38.99 | 34.18 | 33.19 | 33.96 | **33.67** |

**NIQE**

| | Test100 | Rain100H | Rain100L | Test2800 | Test200 | DSEC | **Ours** |
|---|---|---|---|---|---|---|---|
| NIQE | 5.3384 | 11.1536 | 4.4341 | 4.9540 | 4.4178 | 4.1651 | **5.1994** |

We confirmed that the Restormer model successfully detects and removes the rain streaks generated by this pipeline while minimizing damage to the background texture.

---

## Mathematical Background

### Inverse Geometric Warping (1ms 보간)
A 3D point is constructed from the target pixel `(x, y)` and depth `Z`, then reprojected to the source view via the relative transform `T_rel`:

$$
P_{src} = T_{rel} \cdot \left[ \tfrac{x - c_x}{f_x} Z,\ \tfrac{y - c_y}{f_y} Z,\ Z,\ 1 \right]^T
$$

$$
u_{src} = f_x \tfrac{X_{src}}{Z_{src}} + c_x, \quad v_{src} = f_y \tfrac{Y_{src}}{Z_{src}} + c_y
$$

### Marshall–Palmer Raindrop Diameter Distribution
$$
N(D) = N_0 e^{-\Lambda D}, \quad \Lambda = 4.1 R^{-0.21}
$$
Diameters are extracted by inverse-transform sampling:
$$
D = -\tfrac{1}{\Lambda} \ln(1 - U), \quad U \sim \mathcal{U}(0,1)
$$

### Gunn–Kinzer Terminal Velocity
$$
V_t(D) = 9.65 - 10.3\, e^{-0.6 D} \quad [\text{m/s}]
$$

### Sphere Refraction Model (Garg & Nayar, 2006)
The refraction displacement as a function of distance Δ from the raindrop center:
$$
\text{disp}_x = \tfrac{\Delta_x}{r} \left(1 - \tfrac{\sqrt{r^2 - \|\Delta\|^2}}{r}\right) \cdot S_{refr} \cdot \alpha
$$

$$
\alpha = \text{clamp}\left(1 - \tfrac{\|\Delta\|}{r},\ 0\right)^{1.5}
$$

---


### References
- **DSEC**: Gehrig et al., *IEEE RA-L*, 2021
- **Marshall–Palmer**: Marshall & Palmer, *J. Atmos. Sci.*, 1948
- **Gunn–Kinzer**: Gunn & Kinzer, *J. Atmos. Sci.*, 1949
- **MVG**: Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2003
- **Rain refraction**: Garg & Nayar, ACM TOG, 2006
- **v2e**: Hu, Liu & Delbruck, *CVPRW*, 2021

---

## Acknowledgments

This pipeline relies on the following open-source projects.
- [DSEC Dataset](https://dsec.ifi.uzh.ch/) — Input dataset
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
- [v2e](https://github.com/SensorsINI/v2e) — Video-to-Event conversion
