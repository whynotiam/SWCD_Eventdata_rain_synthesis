import numpy as np


class PhysicsEngine:
    def __init__(self, config):
        """
        config.rain_rate:                 rainfall rate R (mm/h)
        config.box_width/height/depth:    dimensions of the 3D box (meters) in which raindrops are generated
        """
        self.rain_rate = config.rain_rate

        # 3D bounding box covering the area in front of the camera
        self.box_width = config.get('box_width', 10.0)
        self.box_height = config.get('box_height', 5.0)
        self.box_depth = config.get('box_depth', 20.0)
        self.volume = self.box_width * (self.box_height * 2) * self.box_depth

        # Marshall-Palmer parameters
        self.N0 = 8000.0  # m^-3 mm^-1
        self.Lambda = 4.1 * (self.rain_rate ** -0.21)
        self.dt = 0.001  # 1ms (1000fps)

        # Particle state
        self.positions = None
        self.diameters = None
        self.velocities = None

        self._initialize_particles()

    def _get_terminal_velocity(self, D):
        """
        Gunn-Kinzer model: physical saturation velocity of large drops under air drag.
        """
        return 9.65 - 10.3 * np.exp(-0.6 * D)

    def _initialize_particles(self):
        """
        Spawn raindrops in 3D space following the Marshall-Palmer distribution.
        """
        print(f"Initializing rain particles... rate: {self.rain_rate} mm/h")

        # Total number of drops = Volume * (N0 / Lambda)
        # (integrate N(D) from 0 to infinity and multiply by volume)
        total_drops_expected = int(0.001 * self.volume * (self.N0 / self.Lambda))
        print(f"Expected raindrop count in volume {self.volume} m^3: {total_drops_expected}")

        # Inverse transform sampling for diameter D.
        # Inverse of the exponential distribution: D = -(1 / Lambda) * ln(1 - U), U ~ Uniform(0,1)
        U = np.random.uniform(0, 1, total_drops_expected)
        self.diameters = -(1.0 / self.Lambda) * np.log(1.0 - U)

        # Filter unrealistic diameters (keep 0.1mm ~ 6.0mm)
        valid_mask = (self.diameters >= 0.1) & (self.diameters <= 6.0)
        self.diameters = self.diameters[valid_mask]
        num_valid = len(self.diameters)

        # Random 3D positions inside the bounding box.
        # Camera frame convention: X (left-right), Y (up-down, +Y = down), Z (forward, +Z = front)
        self.positions = np.zeros((num_valid, 3), dtype=np.float32)
        self.positions[:, 0] = np.random.uniform(-self.box_width / 2, self.box_width / 2, num_valid)
        self.positions[:, 1] = np.random.uniform(-self.box_height, self.box_height, num_valid)
        self.positions[:, 2] = np.random.uniform(0, self.box_depth, num_valid)

        # Terminal velocity (m/s) along +Y (downward).
        # Drops fall at different speeds depending on size (empirical: V = 4.8 * D^0.5).
        self.velocities = np.zeros((num_valid, 3), dtype=np.float32)
        self.velocities[:, 1] = self._get_terminal_velocity(self.diameters)

        print(f"Final particles spawned: {num_valid}")

    def update_particles(self, current_pose, dt=0.001):
        """
        Step the particles forward by 1ms.
        current_pose: {'R': matrix, 't': vector} from DSECSyncManager.
        """
        if not hasattr(self, 'prev_pose'):
            self.prev_pose = current_pose
            return

        # 1) Relative vehicle transform over 1ms
        R_rel = current_pose['R'] @ self.prev_pose['R'].T
        t_rel = current_pose['t'] - (R_rel @ self.prev_pose['t'])

        # 2) Apply gravity-driven fall along Y
        self.positions[:, 1] += self.velocities[:, 1] * self.dt

        # 3) Back-project ego-motion onto particles (keep them in camera coordinates)
        self.positions = (R_rel.T @ (self.positions.T - t_rel.reshape(3, 1))).T

        # 4) Treadmill (periodic boundary conditions).
        # Particles that have passed the camera (Z < 0):
        out_mask_z = self.positions[:, 2] < 0
        if np.any(out_mask_z):
            # Do NOT touch Y here. Preserving the existing fall height keeps vertical density uniform.
            # Shift Z by box_depth (not a hard reset) so inter-particle spacing is preserved
            # and the same rain pattern doesn't repeat (no ghosting).
            self.positions[out_mask_z, 2] += self.box_depth

            # Randomize X to break up repeating patterns.
            num_out = np.sum(out_mask_z)
            self.positions[out_mask_z, 0] = np.random.uniform(-self.box_width / 2, self.box_width / 2, num_out)

        # Particles that have hit the ground (Y > box_height):
        out_mask_y = self.positions[:, 1] > self.box_height
        if np.any(out_mask_y):
            num_out = np.sum(out_mask_y)
            self.positions[out_mask_y, 0] = np.random.uniform(-self.box_width / 2, self.box_width / 2, num_out)
            # Wrap dead drops back to the top of the sky
            self.positions[out_mask_y, 1] -= self.box_height
            self.positions[out_mask_y, 2] = np.random.uniform(0, self.box_depth, num_out)

        self.prev_pose = current_pose

    def get_render_state(self):
        """
        Return the raw post-physics state vector.
        Motion-blur rendering in Phase 3 integrates `vel`.
        """
        return {
            'pos': self.positions,
            'diam': self.diameters,
            'vel': self.velocities,
        }