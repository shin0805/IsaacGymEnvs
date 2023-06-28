import numpy as np
import datetime
import os
import torch

from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Stand(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"] # 1.0
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_heading_weight = self.cfg["env"]["upHeadingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.tilt_weight = self.cfg["env"]["tiltWeight"]
        self.move_forward_weight = self.cfg["env"]["moveForwardWeight"]
        self.height_weight = self.cfg["env"]["heightWeight"]
        self.alive_weight = self.cfg["env"]["aliveWeight"]
        self.simple_progress_weight = self.cfg["env"]["simpleProgressWeight"]
        self.desirable_progress_weight = self.cfg["env"]["desirableProgressWeight"]
        self.omega_weight = self.cfg["env"]["omegaWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.termination_tilt = self.cfg["env"]["terminationTilt"]
        self.termination_up_proj = self.cfg["env"]["terminationUpProj"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numActions"] = 6 
        self.cfg["env"]["numActionHis"] = 4
        self.cfg["env"]["numRotationHis"] = 4

        self.cfg["env"]["numObservations"] = 4 * self.cfg["env"]["numRotationHis"] + self.cfg["env"]["numActions"] * self.cfg["env"]["numActionHis"]
        # self.cfg["env"]["numObservations"] = 9

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_offset = 0 # 20
            cam_pos = gymapi.Vec3(2 + cam_offset, 0.5 + cam_offset, 1.5)
            cam_target = gymapi.Vec3(0.0 + cam_offset, 0.0 + cam_offset, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # sensors_per_env = 4
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        # self.initial_dof_pos = to_torch([-0.1745, 0, -0.1745, 0, -0.1745, 0], device=self.device).repeat((self.num_envs, 1))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        # self.start_rot = self.get_random_quat()
        # self.inv_start_rot = quat_conjugate(self.start_rot)

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([10, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()
        self.torso_pos = to_torch([0, 0, 0], device=self.device).repeat(self.num_envs)
        self.torso_rot = to_torch([0, 0, 0, 1], device=self.device).repeat(self.num_envs)
        self.torso_vel = to_torch([0, 0, 0], device=self.device).repeat(self.num_envs)
        self.omega = to_torch([0, 0, 0], device=self.device).repeat(self.num_envs)
        # self.action_history = to_torch([1, 1, 1, 1, 1, 1], device=self.device).repeat(self.num_envs)
        self.action_history = torch.ones([self.num_envs, self.cfg["env"]["numActionHis"], self.cfg["env"]["numActions"]], device=self.device)
        self.rotation_history = torch.zeros([self.num_envs, self.cfg["env"]["numRotationHis"], 4], device=self.device)
        self.rotation_history[:, :, 3] = 1.0

        self.eval_summary_dir = './rewards/rewards_summaries' + datetime.datetime.now().strftime('_%d-%H-%M-%S')
        # remove the old directory if it exists
        if os.path.exists(self.eval_summary_dir):
            import shutil
            shutil.rmtree(self.eval_summary_dir)
        self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)
        self.frame = 0

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/chair/chair.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        chair_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(chair_asset) # 6
        self.num_bodies = self.gym.get_asset_rigid_body_count(chair_asset) # 7

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(chair_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props] # 0
        self.joint_gears = to_torch(motor_efforts, device=self.device) # 0

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0, self.up_axis_idx))
        # start_pose.r.x = 1.4142 / 2
        # start_pose.r.y = 0
        # start_pose.r.z = 0
        # start_pose.r.w = 1.4142 / 2

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(chair_asset)
        body_names = [self.gym.get_asset_rigid_body_name(chair_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        # extremity_indices = [self.gym.find_asset_rigid_body_index(chair_asset, name) for name in extremity_names]
        # sensor_pose = gymapi.Transform()
        # for body_idx in extremity_indices:
        #     self.gym.create_asset_force_sensor(chair_asset, body_idx, sensor_pose)

        self.chair_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # uvw = torch_rand_float(0, 1.0, (self.num_envs, 3), device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # random rotation
            flag = True
            while flag: 
                uvw = torch_rand_float(0, 1.0, (1, 3), device=self.device)
                start_pose.r.w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
                start_pose.r.x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
                start_pose.r.y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
                start_pose.r.z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
                rand_quat = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device).view(1, 4)
                torso_quat = quat_mul(rand_quat, torch.tensor([0, 0, 0, 1], device=self.device).view(1, 4)).view(1, 4)
                up_vec_proj = get_basis_vector(torso_quat, to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).view(1, 3))[0, 2]
                flag = up_vec_proj < self.termination_up_proj # TODO Not reset at initial rotation

            chair_handle = self.gym.create_actor(env_ptr, chair_asset, start_pose, "chair", i, 1, 0)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, chair_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.chair_handles.append(chair_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, chair_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device) 

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.chair_handles[0], extremity_names[i])

    def get_random_quat(self):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (self.num_envs, 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)
        return new_rot

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], rewards= compute_chair_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.up_heading_weight,
            self.tilt_weight,
            self.move_forward_weight,
            self.height_weight,
            self.alive_weight,
            self.simple_progress_weight,
            self.desirable_progress_weight,
            self.omega_weight,
            self.potentials,
            self.prev_potentials,
            self.torso_pos,
            self.torso_rot,
            self.torso_vel,
            self.omega,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.termination_tilt,
            self.termination_up_proj,
            self.death_cost,
            self.max_episode_length,
            self.cfg["env"]["numRotationHis"],
            self.dummy_obs_buf
        )

        self.frame += 1
        rewards_name = ["up_reward", "heading_reward", "height_reward", "dof_at_limit_cost", "actions_cost"]
        if self.frame % 100 == 0: 
            for i, name in enumerate(rewards_name):
                self.eval_summaries.add_scalar("reward/" + name, rewards[i].mean().item(), self.frame)
            self.eval_summaries.add_scalar("reward/total", self.rew_buf.mean().item(), self.frame)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        #print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.action_history, self.rotation_history, self.torso_pos, self.torso_rot, self.torso_vel, self.omega, self.dummy_obs_buf = compute_chair_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials, self.action_history, self.rotation_history,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions, self.dt,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-1.2, 1.2, (len(env_ids), self.num_dof), device=self.device)
        # rotations = torch.tensor([0, 0, 0, 1.5, 1.5, 1.5], device=self.device).repeat(len(env_ids)).reshape(len(env_ids), self.num_dof)
        # positions_  = to_torch([[0.3, 0.3, 0.3, 3, 3, 3]], device=self.device).repeat(len(env_ids), 1)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        # self.dof_pos[env_ids] = self.initial_dof_pos[env_ids] + positions
        self.dof_vel[env_ids] = velocities

        start_pose = gymapi.Transform()
        uvw = torch_rand_float(0, 1.0, (1, 3), device=self.device)
        start_pose.r.w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        start_pose.r.x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        start_pose.r.y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_chair_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    up_heading_weight,
    tilt_weight,
    move_forward_weight,
    height_weight,
    alive_weight,
    simple_progress_weight,
    desirable_progress_weight,
    omega_weight,
    potentials,
    prev_potentials,
    torso_pos,
    torso_rot,
    torso_vel,
    omega,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    termination_tilt,
    termination_up_proj,
    death_cost,
    max_episode_length,
    numRotationHis,
    dummy_obs_buf):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, int, Tensor) -> Tuple[Tensor, Tensor, List[Tensor]]

    # print(obs_buf[0, :])
    # cost from tilt
    zero_rot = torch.zeros_like(obs_buf[:, 0:4])
    zero_rot[:, 3] = 1.0
    # tilt_cost = torch.norm(obs_buf[:, 0:4] - zero_rot, dim=1) * tilt_weight
    
    # reward from direction headed
    heading_weight_tensor = torch.ones_like(dummy_obs_buf[:, 4]) * heading_weight
    heading_reward = torch.where(dummy_obs_buf[:, 4] > 0.8, heading_weight_tensor, heading_weight * dummy_obs_buf[:, 4] / 0.8)

    # aligning up axis of chair and environment
    # up_reward = torch.zeros_like(heading_reward)
    # up_reward = torch.where(dummy_obs_buf[:, 3] > 0.93, up_reward + up_weight, up_reward + up_weight * dummy_obs_buf[:, 3] / 0.93)
    up_reward = up_weight * torch.min(torch.ones_like(heading_reward), dummy_obs_buf[:, 3] / 0.93)
    # print(dummy_obs_buf[0, 3])

    # reward from direction headed and aligning up axis of chair and environment
    # up_heading_weight_tensor = torch.ones_like(dummy_obs_buf[:, 4]) * up_heading_weight
    # up_heading_reward = torch.where(dummy_obs_buf[:, 3] * dummy_obs_buf[:, 4] > 0.7, up_heading_weight_tensor, up_heading_weight * dummy_obs_buf[:, 3] * dummy_obs_buf[:, 4] / 0.7)

    # up_weight_tensor = torch.ones_like(dummy_obs_buf[:, 3]) * up_weight
    # up_reward = torch.where(dummy_obs_buf[:, 3] > 0.93, up_weight_tensor, up_weight * dummy_obs_buf[:, 3] / 0.93)

    # reward from move forward
    # move_forward_reward = torso_vel[:, 0] * move_forward_weight

    # reward from move height
    # height_reward = torso_pos[:, 2] * dummy_obs_buf[:, 3] * height_weight
    height_reward = torso_pos[:, 2] * height_weight
    height_reward = up_weight * torch.min(torch.ones_like(heading_reward), torso_pos[:, 2] / 0.08)
    # height_reward = torso_pos[:, 2] / (1000 * torch.norm(obs_buf[:, 0:4] - zero_rot, dim=1) + 1) * 1000 * height_weight

    # energy penalty for movement
    # actions_cost = torch.sum(actions ** 2, dim=-1) * actions_cost_scale
    actions_cost = torch.sum((actions - obs_buf[:, 4*numRotationHis+6:4*numRotationHis+12]) ** 2, dim=-1) * actions_cost_scale

    # actions_cost = torch.sum((actions - obs_buf[:, -6:]) ** 2, dim=-1) * actions_cost_scale
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(dummy_obs_buf[:, 5:11]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(dummy_obs_buf[:, 5:11]) > 0.98) * scaled_cost, dim=-1)

    # reward for duration of being alive
    # alive_reward = torch.ones_like(potentials) * alive_weight

    # progress_reward = (potentials - prev_potentials) * dummy_obs_buf[:, 3] * progress_weight
    # simple_progress_reward = torch.zeros_like(heading_reward)
    # simple_progress_reward = (potentials - prev_potentials) * simple_progress_weight
    # desirable_progress_reward = torch.zeros_like(heading_reward)
    # desirable_progress_reward = torch.where((dummy_obs_buf[:, 3] > 0.93) & (torso_pos[:, 2] > 0.055), desirable_progress_weight + (potentials - prev_potentials) * desirable_progress_weight, desirable_progress_reward)
    # desirable_progress_reward = torch.min(torch.ones_like(heading_reward), torch.exp(10 * (dummy_obs_buf[:, 3] / 0.93 - 1))) * \
    #                             torch.min(torch.ones_like(heading_reward), torch.exp(10 * (torso_pos[:, 2] / 0.08 - 1))) * \
    #                             desirable_progress_weight * (potentials - prev_potentials)
    # print(torso_pos[0, 2])
    # print(dummy_obs_buf[0, 3])

    # omega_reward = torch.norm(omega[:, 0:3], dim=1) * torch.norm(obs_buf[:, 0:4] - zero_rot, dim=1) ** 2 * omega_weight

    # total_reward = progress_reward + move_forward_reward + height_reward + alive_reward - tilt_cost - actions_cost + omega_reward
    total_reward = up_reward + heading_reward + height_reward - dof_at_limit_cost - actions_cost
    rewards = [up_reward, heading_reward, height_reward, dof_at_limit_cost, actions_cost]
    # print(f"{tilt_cost[0].item()} , {height_reward[0].item()}, {total_reward[0].item()}")

    # edge_pos
    # x_ofst = 0.095
    # y_ofst = 0.0785
    # lf = torch.zeros_like(torso_pos)
    # lf[:, 0] += x_ofst
    # lf[:, 1] += y_ofst
    # lf_pos = torso_pos + lf
    # lb = torch.zeros_like(torso_pos)
    # lb[:, 0] -= x_ofst
    # lb[:, 1] += y_ofst
    # lb_pos = torso_pos + lb
    # rb = torch.zeros_like(torso_pos)
    # rb[:, 0] -= x_ofst
    # rb[:, 1] -= y_ofst
    # rb_pos = torso_pos + rb
    # rf = torch.zeros_like(torso_pos)
    # rf[:, 0] += x_ofst
    # rf[:, 1] -= y_ofst
    # rf_pos = torso_pos + rf
    # quat = torch.zeros_like(torso_rot)
    # quat[:, 0] = torso_rot[:, 3]
    # quat[:, 1] = torso_rot[:, 0]
    # quat[:, 2] = torso_rot[:, 1]
    # quat[:, 3] = torso_rot[:, 2]
    # edge_pos = torch.matmul(quaternion_to_matrix(quat), torch.stack((lf_pos, lb_pos, rb_pos, rf_pos), dim=2))

    # adjust reward for fallen agents
    # total_reward = torch.where(torch.min(edge_pos[:, 2, :], dim=1).values <= 0.01, torch.ones_like(total_reward) * death_cost, total_reward)
    # total_reward = torch.where(torch.norm(obs_buf[:, 0:4] - zero_rot, dim=1) > termination_tilt, torch.ones_like(total_reward) * death_cost, total_reward)
    # total_reward = torch.where(torso_pos[:, 2] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    # reset = torch.where(torch.min(edge_pos[:, 2, :], dim=1).values <= 0.01, torch.ones_like(reset_buf), reset_buf)
    # reset = torch.where(torch.norm(obs_buf[:, 0:4] - zero_rot, dim=1) > termination_tilt, torch.ones_like(reset_buf), reset) 
    # reset = torch.where(torso_pos[:, 2] < termination_height, torch.ones_like(reset_buf), reset)
    reset = torch.where(dummy_obs_buf[:, 3] < termination_up_proj, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # print(torso_pos[:, 2][0])
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return total_reward, reset, rewards


@torch.jit.script
def compute_chair_observations(obs_buf, root_states, targets, potentials, action_history, rotation_history,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             actions, dt,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    actions_ = actions.view(-1, 1, actions.size()[1])
    action_history = torch.cat((actions_, action_history), dim=1) # 先頭に追加
    action_history = action_history[:, :-1, :] # 末尾を削除

    rotation_ = torso_rotation.view(-1, 1, torso_rotation.size()[1])
    rotation_history = torch.cat((rotation_, rotation_history), dim=1)
    rotation_history = rotation_history[:, :-1, :]

    obs = torch.cat((rotation_history.view(rotation_history.size()[0], -1), action_history.view(action_history.size()[0], -1)), dim=-1)
    # obs = torch.cat((yaw, roll, up_proj.unsqueeze(-1), actions), dim=-1)
    # obs = torch.cat((yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), actions), dim=-1)
    dummy_obs = torch.cat((yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled, actions), dim=-1)

    return obs, potentials, prev_potentials_new, action_history, rotation_history, torso_position, torso_rotation, velocity, ang_velocity, dummy_obs
