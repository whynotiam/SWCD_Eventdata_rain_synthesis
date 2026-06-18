# SWCD_Eventdata_rain_synthesis

# Physics-Based 3D Event-RGB Rain Synthesis for DSEC

---

## 📌 Overview

본 프로젝트는 자율주행용 Event–RGB 데이터셋인 [**DSEC**](https://dsec.ifi.uzh.ch/)에 대해, **물리 법칙(Marshall–Palmer, Gunn–Kinzer)**에 기반한 3D 우천(rain) 합성 파이프라인을 제공합니다. 단순한 2D 빗줄기 overlay가 아닌, 다음을 동시에 보장하는 합성 데이터를 생성합니다.

- **3D 공간 상에서의 빗방울 시뮬레이션** — 카메라 포즈(R, t)와 Z-buffer를 활용한 실제 깊이 정보 기반 생성
- **1ms 단위(1000Hz) 시공간 정밀 동기화** — 20Hz 원본 RGB를 inverse geometric warping으로 보간
- **Event–RGB 페어의 물리적 정합성** — 동일 물리 연산을 이원화된 렌더링 스트림에 통과시켜 RGB frame과 event mask GT를 동시 생성
- **3D Occlusion 처리** — LiDAR 기반 depth map으로 건물·차량 뒤의 빗방울을 자연스럽게 가림

생성된 데이터는 event 기반 **Deraining** 모델 학습용으로 활용할 수 있으며, 사전학습된 Restormer를 통한 벤치마킹에서 PSNR **33.67dB**, NIQE **5.1994**의 결과를 달성했습니다.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Physical particle generation** | Marshall–Palmer 분포로 빗방울 직경 샘플링, Gunn–Kinzer 모델로 종단속도(terminal velocity) 부여 |
| **Spatio-temporal sync (1ms)** | SLERP + linear interpolation으로 카메라 포즈를 1ms 단위로 재구성, inverse geometric warping으로 RGB 보간 |
| **Z-buffer occlusion** | DSEC LiDAR disparity → depth 변환 후 픽셀 단위 가림막 처리 |
| **Dual-stream rendering** | RGB stream(워핑 배경 + 모션블러)과 Event stream(static noise canvas + v2e)을 동시 출력 |
| **Optical realism** | Sphere refraction, edge attenuation, optical bloom, atmospheric veiling, sensor noise |
| **Treadmill recycling** | Ego-motion 반영 후 화면 이탈 입자 재배치로 long-term 안정성 확보 |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        [ Phase 0: Preprocessing ]                       │
│                                                                         │
│   DSEC raw data                                                         │
│     ├── disparity (.png 16-bit)  ──►  preprocess_depth.py  ──► .npy     │
│     └── images/left/             ──►  COLMAP  ──► sync_poses.py        │
│                                                       │                 │
│                                                       ▼                 │
│                                              poses_colmap.csv          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       [ Phase 1: Spatio-Temporal Sync ]                 │
│                                                                         │
│   DSECSyncManager  (sync_manager.py)                                    │
│     • SLERP rotation interpolation @ 1ms                                │
│     • Linear position interpolation @ 1ms                               │
│     • Nearest neighbor depth/RGB lookup + LRU cache                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  [ Phase 2: Physics & Spatial Middleware ]              │
│                                                                         │
│   PhysicsEngine   (physics_engine.py)                                   │
│     • Marshall–Palmer:   N(D) = N₀ exp(-ΛD),  Λ = 4.1 R⁻⁰·²¹            │
│     • Gunn–Kinzer:       Vt(D) = 9.65 - 10.3 exp(-0.6D) [m/s]           │
│     • Ego-motion 역투영을 통한 카메라 좌표계 입자 유지                  │
│                                                                         │
│   SpatialMiddleware  (spatial_middleware.py)                            │
│     • Pinhole 3D → 2D 투영 (sub-pixel float32 유지)                    │
│     • Frustum culling + Z-buffer occlusion                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  [ Phase 3: Dual-Stream Rendering ]                     │
│                                                                         │
│   RainRenderer  (renderer.py)                                           │
│                                                                         │
│   ┌─ Stream A (RGB) ──────────────────────────────────────────┐         │
│   │  warped_bg ⊕ refraction ⊕ streak ⊕ bloom ⊕ fog ⊕ noise    │         │
│   │  → 20ms 노출 시간 동안 1ms 프레임 적분 (motion blur)      │         │
│   │  → rgb_frames/frame_XXXXXX.png                            │         │
│   └───────────────────────────────────────────────────────────┘         │
│                                                                         │
│   ┌─ Stream B (Event GT) ─────────────────────────────────────┐         │
│   │  static_noise_canvas ⊕ pure rain streak                   │         │
│   │  → v2e EventEmulator (streaming to .h5)                   │         │
│   │  → events_synthetic.h5                                    │         │
│   └───────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     [ Phase 4: Event Merging ]                          │
│                                                                         │
│   merge_events.py                                                       │
│     • 원본 DSEC events.h5 + synthetic rain events                       │
│     • 절대 타임스탬프 기준 정렬, ms_to_idx 재계산                       │
│     • DSEC 공식 포맷 (.h5)으로 InputEvent.h5 출력                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (RTX 3090 24GB 기준 검증 완료)
- **RAM**: 32GB 이상 권장
- **Disk**: 시퀀스당 약 50–100GB의 출력 공간

### Software
- Python ≥ 3.8
- CUDA ≥ 11.3
- COLMAP ≥ 3.7 (CLI 사용)

### Python Dependencies
```bash
pip install numpy pandas scipy
pip install torch torchvision      # CUDA 빌드
pip install opencv-python
pip install pycolmap
pip install h5py hdf5plugin
pip install PyYAML
```

### Third-party
- **[v2e](https://github.com/SensorsINI/v2e)** — `third_party/v2e/` clone 
  ```bash
  mkdir -p third_party && cd third_party
  git clone https://github.com/SensorsINI/v2e.git
  ```

---

## 📂 Repository Structure

```
Rain_Generator/
├── config.py                      # 동적 경로 / 카메라·물리·렌더링 파라미터
├── sync_manager.py                # 1ms 단위 포즈·Depth·RGB 동기화 매니저
├── physics_engine.py              # Marshall–Palmer + Gunn–Kinzer 물리엔진
├── spatial_middleware.py          # 3D→2D 투영 / Frustum culling / Z-buffer
├── renderer.py                    # 굴절·모션블러·블룸·노이즈 듀얼 렌더러
├── main_parallel.py               # 전체 파이프라인 오케스트레이터 (entrypoint)
│
├── sync_poses.py                  # COLMAP 결과 → poses_colmap.csv 추출·보간
├── preprocess_depth.py            # disparity(.png) → depth(.npy) 변환
├── merge_events.py                # 원본 + 합성 이벤트 병합 → InputEvent.h5
│
├── utils/
│   └── coordinate_utils.py        # 상대 변환 행렬 T_rel 계산
│
├── third_party/
│   └── v2e/                       # SensorsINI/v2e (clone 필요)
│
├── colmap_workspace/              # COLMAP sparse reconstruction 결과
│   └── colmap_results_<JOB_ID>/sparse/0/
│
└── data/
    ├── DSEC/<seq_name>/           # 입력 데이터
    │   ├── images/left/rectified/*.png
    │   ├── images/left/image_timestamps.txt
    │   ├── events/left/events.h5
    │   ├── ground_truth/disparity/*.png
    │   ├── ground_truth/disparity_timestamps.txt
    │   ├── ground_truth/poses_colmap.csv     # ← sync_poses.py 산출물
    │   ├── lidar/event_cam_coords/*.npy      # ← preprocess_depth.py 산출물
    │   └── calibration/cam_to_cam.yaml
    │
    └── synthetic_output/<seq_name>/<seq_name>/   # 출력 데이터
        ├── rgb_frames/frame_XXXXXX.png      # 모션블러 적용 RGB (50ms 단위)
        ├── gt_masks/mask_XXXXXX.png         # 순수 rain GT mask
        ├── events_synthetic.h5              # RDE (Rain Drop Events)
        └── final_input/InputEvent.h5        # 원본+합성 병합 결과
```

---

## 🚀 Usage

### 0. 환경 변수 설정 (권장)
모든 스크립트가 `WORKSPACE_DIR`을 인식합니다.
```bash
export WORKSPACE_DIR="/path/to/Rain_Generator"
```

### 1. DSEC 데이터 준비

[DSEC 공식 홈페이지](https://dsec.ifi.uzh.ch/)에서 다음 데이터를 다운로드 후 `data/DSEC/<seq_name>/` 하위에 배치합니다 (예시: `zurich_city_04_a`).

| 항목 | 경로 |
|---|---|
| Rectified left images | `images/left/rectified/*.png` |
| Image timestamps | `images/left/image_timestamps.txt` |
| Events | `events/left/events.h5` |
| Disparity | `ground_truth/disparity/*.png` |
| Disparity timestamps | `ground_truth/disparity_timestamps.txt` |
| Calibration | `calibration/cam_to_cam.yaml` |

### 2. Depth Map 전처리 (Disparity → Depth)

`cam_to_cam.yaml`의 `disparity_to_depth.cams_12` 매트릭스에서 `fx · baseline`을 자동 파싱하여 16-bit disparity PNG를 미터 단위 depth `.npy`로 변환합니다. Morphological closing을 통한 hole filling과 하늘 영역(Depth=0) → 1000m 처리가 포함됩니다.

```bash
python preprocess_depth.py --seq_name zurich_city_04_a
```
출력: `data/DSEC/<seq_name>/lidar/event_cam_coords/*.npy`

### 3. COLMAP 기반 카메라 포즈 추출

DSEC은 IMU/GNSS는 제공하지만 정밀 카메라 포즈는 제공하지 않으므로, **COLMAP**으로 sparse reconstruction을 수행합니다.

```bash
# (예시) COLMAP CLI 자동화
colmap automatic_reconstructor \
    --workspace_path  ./colmap_workspace/colmap_results_<JOB_ID> \
    --image_path      ./data/DSEC/<seq_name>/images/left/rectified \
    --single_camera   1
```

이후 reconstruction 결과(sparse/0)에서 포즈를 추출하고, 빠진 프레임을 SLERP + linear interpolation으로 보간하여 모든 RGB 타임스탬프에 대한 포즈를 만듭니다.

```bash
python sync_poses.py \
    --seq_name zurich_city_04_a \
    --colmap_dir ./colmap_workspace/colmap_results_<JOB_ID>/sparse/0
```
출력: `data/DSEC/<seq_name>/ground_truth/poses_colmap.csv`
(컬럼: `ts, x, y, z, qx, qy, qz, qw`)

> **호환성 참고**: `sync_poses.py`는 `image.qvec`(구버전)과 `image.cam_from_world`(신버전) 양쪽 pycolmap API를 모두 자동 감지하여 처리합니다.

### 4. 메인 파이프라인 실행 (RGB + Event 합성)

```bash
python main_parallel.py \
    --seq_name zurich_city_04_a \
    --start 0      \
    --end   7000      # 7초 구간 (ms 단위, 생략 시 시퀀스 전체)
```

출력물:
- `data/synthetic_output/<seq_name>/<seq_name>/rgb_frames/frame_XXXXXX.png` — 모션블러 적용 합성 RGB (50ms 간격)
- `data/synthetic_output/<seq_name>/<seq_name>/gt_masks/mask_XXXXXX.png` — 픽셀 단위 rain GT mask
- `data/synthetic_output/<seq_name>/<seq_name>/events_synthetic.h5` — RDE (rain drop events)

### 5. 원본 DSEC 이벤트와 합성 이벤트 병합

DSEC 표준 포맷(`x`, `y`, `t`, `p`, `t_offset`, `ms_to_idx`)으로 병합합니다. 원본의 `t_offset`을 상속하므로 타임라인이 정확히 일치합니다.

```bash
python merge_events.py \
    --job_id  <JOB_ID> \
    --seq_name zurich_city_04_a
```
출력: `data/synthetic_output/<seq_name>/<seq_name>/final_input/InputEvent.h5`

---

## ⚙️ Configuration

모든 하이퍼파라미터는 [`config.py`](config.py)에서 한 곳에서 관리합니다. `cam_to_cam.yaml`에서 카메라 매트릭스를 동적으로 로드합니다.

### 주요 파라미터

| 카테고리 | 파라미터 | 기본값 | 설명 |
|---|---|---|---|
| **Physics** | `rain_rate` | `10.0` mm/h | 강우량 (R) |
| | `box_width / height / depth` | `50 / 30 / 30` m | 빗방울 생성 3D Bounding Box |
| **Render** | `exposure_time_ms` | `20` | 모션블러 적분 노출 시간 |
| | `refraction_strength` | `1.8` | 빗방울 굴절 강도 (S_refr) |
| | `specular_intensity` | `0.2` | 빗방울 표면 반사광 |
| **v2e** | `v2e_pos_thres / neg_thres` | `0.2` | 이벤트 발화 임계값 |
| | `v2e_sigma_thres` | `0.0` | RDE 순수 마스크용 (노이즈 차단) |
| | `v2e_cutoff_hz / leak_rate / shot_noise` | `0` | 노이즈 관련 모두 비활성화 |

### `renderer.py` 내부 시각 효과

| 효과 | 변수 | 기본값 |
|---|---|---|
| Rain opacity | `rain_opacity` | `0.5` |
| 빗줄기 stretch (Y축) | `stretch_y` | `1000.0` |
| 빗방울 최대 두께 | `clamp(d, max=8.0)` | `8 px` |
| Bloom threshold | `bloom_threshold` | `0.75` |
| Bloom intensity | `bloom_intensity` | `0.6` |
| 조도 감쇠 (악천후) | `illumination_factor` | `0.88` |
| 물안개 강도 | `fog_density` | `0.10` |
| 센서 노이즈 σ | `noise_std_dev` | `0.02` |

---

## 📊 Results

DSEC의 주간/야간 시퀀스, 50ms 단위 7초 frame의 RGB–GT 쌍에서 평가 (GPU: RTX 3090 ×1).

### 물리 특성 검증
- 빗방울의 픽셀 길이는 종단속도와 카메라 거리의 역수에 비례 → **부채꼴 형태 분포** 확인 (논문 그림 1.b, 1.c)
- Marshall–Palmer 직경 샘플링과 깊이의 독립성 검증 완료

### Event–RGB 동기화 Precision
- 2ms 윈도우 내 이벤트 중 rain mask 영역에 포함된 비율: **0.6337**

### Restormer 벤치마킹 (Deraining)

**PSNR (dB)**

| | Test100 | Rain100H | Rain100L | Test2800 | Test200 | Average | **Ours** |
|---|---|---|---|---|---|---|---|
| Restormer | 32.00 | 31.46 | 38.99 | 34.18 | 33.19 | 33.96 | **33.67** |

**NIQE**

| | Test100 | Rain100H | Rain100L | Test2800 | Test200 | DSEC | **Ours** |
|---|---|---|---|---|---|---|---|
| NIQE | 5.3384 | 11.1536 | 4.4341 | 4.9540 | 4.4178 | 4.1651 | **5.1994** |

Restormer 모델이 본 파이프라인 합성 빗줄기를 성공적으로 탐지·제거하면서도 배경 텍스처 손상을 최소화하는 것을 확인

---

## 🧮 Mathematical Background

### Inverse Geometric Warping (1ms 보간)
타겟 픽셀 `(x, y)`와 깊이 `Z`로 3D 점을 만들고, 상대 변환 `T_rel`을 거쳐 source view로 재투영:

$$
P_{src} = T_{rel} \cdot \left[ \tfrac{x - c_x}{f_x} Z,\ \tfrac{y - c_y}{f_y} Z,\ Z,\ 1 \right]^T
$$

$$
u_{src} = f_x \tfrac{X_{src}}{Z_{src}} + c_x, \quad v_{src} = f_y \tfrac{Y_{src}}{Z_{src}} + c_y
$$

### Marshall–Palmer 빗방울 직경 분포
$$
N(D) = N_0 e^{-\Lambda D}, \quad \Lambda = 4.1 R^{-0.21}
$$
역변환 샘플링으로 빗방울 직경 추출:
$$
D = -\tfrac{1}{\Lambda} \ln(1 - U), \quad U \sim \mathcal{U}(0,1)
$$

### Gunn–Kinzer 종단속도
$$
V_t(D) = 9.65 - 10.3\, e^{-0.6 D} \quad [\text{m/s}]
$$

### Sphere Refraction Model (Garg & Nayar, 2006)
빗방울 중심으로부터의 변위 Δ에 대한 굴절 displacement:
$$
\text{disp}_x = \tfrac{\Delta_x}{r} \left(1 - \tfrac{\sqrt{r^2 - \|\Delta\|^2}}{r}\right) \cdot S_{refr} \cdot \alpha
$$
$$
\alpha = \text{clamp}\left(1 - \tfrac{\|\Delta\|}{r},\ 0\right)^{1.5}
$$

---


### 관련 참고문헌
- **DSEC**: Gehrig et al., *IEEE RA-L*, 2021
- **Marshall–Palmer**: Marshall & Palmer, *J. Atmos. Sci.*, 1948
- **Gunn–Kinzer**: Gunn & Kinzer, *J. Atmos. Sci.*, 1949
- **MVG**: Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2003
- **Rain refraction**: Garg & Nayar, ACM TOG, 2006
- **v2e**: Hu, Liu & Delbruck, *CVPRW*, 2021

---

## 🙏 Acknowledgments

본 파이프라인은 다음 오픈소스 프로젝트에 의존합니다.
- [DSEC Dataset](https://dsec.ifi.uzh.ch/) — 입력 데이터셋
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
- [v2e](https://github.com/SensorsINI/v2e) — Video-to-Event 변환
