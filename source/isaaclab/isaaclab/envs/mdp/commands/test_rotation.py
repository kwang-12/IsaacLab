import torch
import isaaclab.utils.math as math_utils

def sample_quat_within_rot_angle(center_q: torch.Tensor, max_angle: torch.Tensor, num_samples: int = 1):
    """
    Sample unit quaternions within max_angle (radians) of center_q.
    center_q: (..., 4) tensor of reference quaternions (unit norm)
    max_angle: scalar max angular distance in radians
    num_samples: how many samples to generate per center

    Returns: (num_samples, 4) tensor of sampled quaternions
    """
    device = center_q.device
    shape = (num_samples,)

    # Convert angle to cosine of half-angle for quaternion dot-product space
    cos_theta = torch.cos(max_angle / 2.0)

    # Sample cos(ϕ) uniformly in [cosθ, 1]
    u = torch.rand(shape, device=device)
    cos_phi = (1.0 - u) + u * cos_theta  # Uniform in [cosθ, 1]
    sin_phi = torch.sqrt(1.0 - cos_phi**2)

    # Uniform direction on the 3D "equator" orthogonal to the center quaternion
    v = torch.randn((num_samples, 3), device=device)
    v = v / v.norm(dim=-1, keepdim=True)

    # Build local-frame quaternion (cosφ, sinφ·v̂)
    local_quat = torch.cat([cos_phi.unsqueeze(-1), sin_phi.unsqueeze(-1) * v], dim=-1)  # shape: (N, 4)

    # Rotate local quaternion into center frame
    rotated_quat = math_utils.quat_mul(center_q.expand(num_samples, 4), local_quat)
    return rotated_quat

center_q = torch.tensor([[1,0,0,0],[1,0,0,0]])
max_angle = 10/180*torch.pi
num_samples = 3

rotated_q = sample_quat_within_rot_angle(center_q[0,:], torch.tensor(max_angle), num_samples)

euler_center_q = math_utils.euler_xyz_from_quat(center_q)
euler_rotated_q = math_utils.euler_xyz_from_quat(rotated_q)

axis_angle_center_q = math_utils.axis_angle_from_quat(center_q)
axis_angle_rotated_q = math_utils.axis_angle_from_quat(rotated_q)

print(rotated_q)
print(euler_center_q)
print(euler_rotated_q)
print(axis_angle_center_q)
print(axis_angle_rotated_q)