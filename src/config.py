import os
import yaml

# Project root auto-detection: this file lives in <repo>/src/, so the workspace
# is one directory up. WORKSPACE_DIR environment variable overrides this default.
_DEFAULT_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    def __init__(self, seq_name="zurich_city_04_a"):
        # Paths
        self.workspace_dir = os.environ.get("WORKSPACE_DIR", _DEFAULT_WORKSPACE)

        self.seq_name = seq_name
        self.base_path = os.path.join(self.workspace_dir, "data", "DSEC", self.seq_name)
        self.output_dir = os.path.join(self.workspace_dir, "data", "synthetic_output", self.seq_name)
        self.rgb_out_dir = os.path.join(self.output_dir, "rgb_blurred")

        # Load DSEC camRect1 (Left RGB) intrinsics dynamically from YAML
        calib_file = os.path.join(self.base_path, "calibration", "cam_to_cam.yaml")

        if not os.path.exists(calib_file):
            raise FileNotFoundError(
                f"Calibration file not found: {calib_file}\n"
                f"Please make sure the DSEC sequence '{self.seq_name}' is placed under "
                f"{os.path.join(self.workspace_dir, 'data', 'DSEC')}"
            )

        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)

        # Extract camRect1 camera matrix [fx, fy, cx, cy] and resolution [width, height]
        cam_matrix = calib_data['intrinsics']['camRect1']['camera_matrix']
        resolution = calib_data['intrinsics']['camRect1']['resolution']

        self.cam = {
            'fx': float(cam_matrix[0]),
            'fy': float(cam_matrix[1]),
            'cx': float(cam_matrix[2]),
            'cy': float(cam_matrix[3]),
            'width': int(resolution[0]),
            'height': int(resolution[1]),
        }

        print(f"[{self.seq_name}] Camera parameters loaded: {self.cam['width']}x{self.cam['height']}")

        # Physics Engine parameters
        self.rain_rate = 10.0   # Rainfall rate (mm/h)
        self.box_width = 50.0   # Virtual box width (m)
        self.box_height = 30.0  # Virtual box height (m)
        self.box_depth = 30.0   # Virtual box depth (m)

        # Renderer options
        self.exposure_time_ms = 20      # Exposure time for motion blur
        self.refraction_strength = 1.8  # Background refraction strength
        self.specular_intensity = 0.2   # Raindrop surface specular intensity

        # v2e (Event Emulator) parameters
        # Noise-related options are zeroed out to extract pure RDE masks.
        self.v2e_pos_thres = 0.2
        self.v2e_neg_thres = 0.2
        self.v2e_sigma_thres = 0.0
        self.v2e_cutoff_hz = 0
        self.v2e_leak_rate_hz = 0.0
        self.v2e_shot_noise_rate_hz = 0.0

    def get(self, key, default=None):
        return getattr(self, key, default)