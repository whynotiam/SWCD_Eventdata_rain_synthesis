import torch

def get_relative_transform(R_curr, t_curr, R_src, t_src, device):
    
    R_c = torch.tensor(R_curr, dtype=torch.float32, device=device)
    t_c = torch.tensor(t_curr, dtype=torch.float32, device=device).view(3, 1)
    R_s = torch.tensor(R_src, dtype=torch.float32, device=device)
    t_s = torch.tensor(t_src, dtype=torch.float32, device=device).view(3, 1)

    # P_src = R_s^T * (R_c * P_curr + t_c - t_s)
    R_rel = R_s.T @ R_c
    t_rel = R_s.T @ (t_c - t_s)
    
    T_rel = torch.eye(4, device=device)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3:] = t_rel
    return T_rel