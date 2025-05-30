import torch
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

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

def test_dict(
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]]):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "max_radian"]]
    print(range_list)
    ranges = torch.tensor(range_list)
    print(ranges)
    rand_samples = math_utils.sample_uniform(ranges[:,0], ranges[:,1], (len(env_ids),4),device=env_ids.device)
    print(rand_samples)

env_ids = torch.Tensor([0,1,2])
pose_range = {
    "x": (-1.0, 1.0),
    "y": (-2.0, 2.0),
    "z": (-3.0, 3.0),
    "max_radian": (-0.1, 0.1)
}
test_dict(env_ids, pose_range)

def sample_quat_within_angle(center_q: torch.Tensor, max_angle: float):
    """
    Sample unit quaternions within max_angle (radians) of center_q.
    center_q: (..., 4) tensor of reference quaternions (unit norm)
    max_angle: scalar max angular distance in radians
    num_samples: how many samples to generate per center

    Returns: (num_samples, 4) tensor of sampled quaternions
    """
    # Sample cos(ϕ) uniformly in [cosθ, 1]
    u = torch.rand(center_q.shape[0], device=center_q.device)
    cos_phi = (1.0 - u) + u * torch.cos(torch.tensor(max_angle, device=center_q.device) / 2.0)  # Uniform in [cosθ, 1]
    sin_phi = torch.sqrt(1.0 - cos_phi**2)

    # Uniform direction on the 3D "equator" orthogonal to the center quaternion
    v = torch.randn((center_q.shape[0], 3), device=center_q.device)
    v = v / v.norm(dim=-1, keepdim=True)

    # Build local-frame quaternion (cosφ, sinφ·v̂)
    local_quat = torch.cat([cos_phi.unsqueeze(-1), sin_phi.unsqueeze(-1) * v], dim=-1)  # shape: (N, 4)

    # Rotate local quaternion into center frame
    rotated_quat = math_utils.quat_mul(center_q, local_quat)
    return rotated_quat

def reset_root_state_clamped_angular_distance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    max_radian: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    # extract ranges
    xyz_pos_ranges = torch.tensor([position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]])
    xyz_vel_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z",
                                                                                   "roll", "pitch", "yaw"]])
    # generate randomized root pose
    xyz_pos_rand_samples = math_utils.sample_uniform(xyz_pos_ranges[:,0], xyz_pos_ranges[:,1], 
                                                     (len(env_ids),3), device=env_ids.device)
    orientations = sample_quat_within_angle(root_states[:, 3:7], max_radian)
    # generate randomized root velocity
    xyzrpy_vel_rand_samples = math_utils.sample_uniform(xyz_pos_ranges[:,0], xyz_pos_ranges[:,1], 
                                                        (len(env_ids),6), device=env_ids.device)
    # assign new pose
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + xyz_pos_rand_samples
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    # assign new velocities
    velocities = root_states[:, 7:13] + xyzrpy_vel_rand_samples
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    