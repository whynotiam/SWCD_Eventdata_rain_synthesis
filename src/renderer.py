import torch
import torch.nn.functional as F


class RainRenderer:
    def __init__(self, config):
        self.width = config.cam['width']
        self.height = config.cam['height']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.height),
            torch.linspace(-1, 1, self.width),
            indexing='ij',
        )
        self.base_grid = torch.stack((x, y), dim=-1).unsqueeze(0).to(self.device)

        self.refraction_strength = config.refraction_strength

        # Visual shape parameters for rain streaks
        self.rain_opacity = 0.5
        self.stretch_y = 1000.0

    def render(self, bg_tensor, drop_states):
        n0 = 0 if drop_states is None else len(drop_states['u'])

        if drop_states is None or len(drop_states['u']) == 0:
            print(f"[RENDER] drop_states empty (0 from spatial stage)")
            return bg_tensor, torch.zeros((1, 1, self.height, self.width), device=self.device)

        disp_map = torch.zeros((1, 2, self.height, self.width), device=self.device)
        streak_map = torch.zeros((1, 1, self.height, self.width), device=self.device)

        u = torch.tensor(drop_states['u'], dtype=torch.float32, device=self.device)
        v = torch.tensor(drop_states['v'], dtype=torch.float32, device=self.device)
        d = torch.tensor(drop_states['diam_pix'], dtype=torch.float32, device=self.device)
        z = torch.tensor(drop_states['z'], dtype=torch.float32, device=self.device)

        # Drop "giant ghost" raindrops directly in front of the lens (Z < 0.2m)
        valid_z = z > 0.2
        u, v, d, z = u[valid_z], v[valid_z], d[valid_z], z[valid_z]
        n2 = len(u)

        # Cap drop thickness so very close drops don't blow up beyond a few pixels
        d = torch.clamp(d, max=8.0)

        # If nothing survives culling, return the original background
        if len(u) == 0:
            print(f"[RENDER] 0 after culling | input:{n0} after z>0.2:{n2}")
            return bg_tensor, torch.zeros((1, 1, self.height, self.width), device=self.device)

        # Painter's algorithm: render back-to-front
        sorted_indices = torch.argsort(z, descending=True)
        u, v, d = u[sorted_indices], v[sorted_indices], d[sorted_indices]

        chunk_size = 10000
        for start_idx in range(0, len(u), chunk_size):
            end_idx = min(start_idx + chunk_size, len(u))
            u_c = u[start_idx:end_idx]
            v_c = v[start_idx:end_idx]
            d_c = d[start_idx:end_idx]

            r_c = d_c / 2
            r_int_c = torch.ceil(r_c).int()

            valid_drops = r_int_c >= 1
            if not valid_drops.any():
                continue

            u_c, v_c, r_c, r_int_c = u_c[valid_drops], v_c[valid_drops], r_c[valid_drops], r_int_c[valid_drops]

            R_x_max = r_int_c.max().item()
            R_y_max = min(int(R_x_max * self.stretch_y) + 1, self.height)

            dy, dx = torch.meshgrid(
                torch.arange(-R_y_max, R_y_max + 1, device=self.device),
                torch.arange(-R_x_max, R_x_max + 1, device=self.device),
                indexing='ij',
            )
            dy, dx = dy.flatten(), dx.flatten()

            u_i = u_c.int().unsqueeze(1)
            v_i = v_c.int().unsqueeze(1)

            xx = u_i + dx.unsqueeze(0)
            yy = v_i + dy.unsqueeze(0)

            dist_sq = (dx.unsqueeze(0)) ** 2 + (dy.unsqueeze(0) / self.stretch_y) ** 2
            dist_sq = dist_sq.expand(len(u_c), -1)

            in_circle = dist_sq < (r_c ** 2).unsqueeze(1)
            in_global_bounds = (xx >= 0) & (xx < self.width) & (yy >= 0) & (yy < self.height)

            mask = in_global_bounds & in_circle
            valid_mask = mask.reshape(-1)

            if not valid_mask.any():
                continue

            xx_v = xx.reshape(-1)[valid_mask].long()
            yy_v = yy.reshape(-1)[valid_mask].long()
            dist_sq_v = dist_sq.reshape(-1)[valid_mask]

            r_v = r_c.unsqueeze(1).expand(-1, len(dx)).reshape(-1)[valid_mask]
            dx_v = dx.unsqueeze(0).expand(len(u_c), -1).reshape(-1)[valid_mask]
            dy_v = dy.unsqueeze(0).expand(len(u_c), -1).reshape(-1)[valid_mask]

            # Linear edge attenuation (^1.5) so opacity falls cleanly to 0 at the rim
            norm_dist = torch.sqrt(dist_sq_v) / r_v
            alpha = torch.clamp(1.0 - norm_dist, min=0.0) ** 1.5

            z_val = torch.sqrt(torch.clamp(r_v ** 2 - dist_sq_v, min=1e-6))

            grad_x = dx_v / r_v
            grad_y = (dy_v / self.stretch_y) / r_v

            val_x = grad_x * (1 - z_val / r_v) * self.refraction_strength * alpha
            val_y = grad_y * (1 - z_val / r_v) * self.refraction_strength * alpha

            streak_intensity = (1.0 - norm_dist) * self.rain_opacity * alpha

            flat_idx = yy_v * self.width + xx_v

            disp_map.reshape(2, -1)[0].scatter_(0, flat_idx, val_x)
            disp_map.reshape(2, -1)[1].scatter_(0, flat_idx, val_y)

            current_streak = streak_map.reshape(-1)
            new_streak = torch.max(current_streak[flat_idx], streak_intensity)
            current_streak.scatter_(0, flat_idx, new_streak)

        disp_grid = disp_map.permute(0, 2, 3, 1)
        disp_grid[..., 0] /= (self.width / 2)
        disp_grid[..., 1] /= (self.height / 2)

        warped_grid = torch.clamp(self.base_grid + disp_grid, -1, 1)
        refracted_bg = F.grid_sample(bg_tensor, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)

        final_rgb = refracted_bg + streak_map
        final_rgb = torch.clamp(final_rgb, 0, 1)

        # ============================================================
        # Optical Scattering (Bloom)
        # Models light blooming where bright highlights (streetlights, headlights)
        # interact with raindrops.
        # ============================================================
        bloom_threshold = 0.75  # only pixels brighter than this emit bloom (0.0 - 1.0)
        bloom_intensity = 0.6   # strength of the bloom layer

        # 1) Extract bright highlight regions
        bright_regions = torch.clamp(final_rgb - bloom_threshold, min=0.0) / (1.0 - bloom_threshold)

        # 2) Fast GPU blur (two box blurs ~ Gaussian approximation).
        # Larger kernel_size spreads the bloom wider.
        kernel_size = 15
        pad = kernel_size // 2
        blurred_bright = F.avg_pool2d(bright_regions, kernel_size=kernel_size, stride=1, padding=pad)
        blurred_bright = F.avg_pool2d(blurred_bright, kernel_size=kernel_size, stride=1, padding=pad)

        # 3) Screen-blend the bloom layer with the base (softer than plain addition)
        bloom_layer = blurred_bright * bloom_intensity
        final_rgb = final_rgb + bloom_layer - (final_rgb * bloom_layer)
        final_rgb = torch.clamp(final_rgb, 0, 1)

        # ============================================================
        # Global Degradation (rainy-day realism / PSNR target)
        # ============================================================

        # 1) Darkening: lower scene illumination under bad weather (0.8 - 0.9)
        illumination_factor = 0.88
        final_rgb = final_rgb * illumination_factor

        # 2) Atmospheric veiling: reduce global contrast.
        # Higher fog_density yields a hazier look and lower PSNR (recommend 0.05 - 0.2).
        fog_density = 0.10
        fog_color = torch.tensor([0.8, 0.8, 0.8], device=self.device).view(1, 3, 1, 1)
        final_rgb = final_rgb * (1 - fog_density) + (fog_color * fog_density)

        # 3) Sensor noise (camera noise under bad weather; recommend 0.01 - 0.03)
        noise_std_dev = 0.02
        noise = torch.randn_like(final_rgb) * noise_std_dev
        final_rgb = torch.clamp(final_rgb + noise, 0, 1)

        # Pure binary rain mask: 1 where rain is present, 0 elsewhere
        pure_mask = (streak_map > 0.0).float()

        return torch.clamp(final_rgb, 0, 1), pure_mask