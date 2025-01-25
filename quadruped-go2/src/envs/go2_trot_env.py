"""Policy outputs desired CoM speed for Go2 to track the desired speed."""

import itertools
import time
from collections import deque
from typing import Sequence

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.envs.robots.modules.controller import raibert_swing_leg_controller, qp_torque_optimizer
from src.envs.robots.modules.gait_generator import phase_gait_generator
from src.envs.robots import go2_robot, go2
from src.envs.robots.modules.planner.path_planner import PathPlanner
from src.envs.robots.motors import MotorControlMode, concatenate_motor_actions
from src.envs.terrains.wild_env import WildTerrainEnv
from src.ha_teacher.ha_teacher import HATeacher
from src.coordinator.coordinator import Coordinator
from src.physical_design import MATRIX_P
from omegaconf import DictConfig

from src.utils.utils import ActionMode


def generate_seed_sequence(seed, num_seeds):
    np.random.seed(seed)
    return np.random.randint(0, 100, size=num_seeds)


random_push_sequence = generate_seed_sequence(seed=1, num_seeds=100)
dual_push_sequence = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
push_magnitude_list = [-1.0, 1.2, -1.3, 1.4, -1.5]

# For backwards
# push_delta_vel_list_x = [-0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25]
# push_delta_vel_list_y = [-0.62, 1.25, -0.7, 0.6, -0.55, 0.6, -0.6]
# push_delta_vel_list_z = [-0.72, -0.72, -0.72, -0.72, -0.72, -0.72, -0.72]
# push_interval = np.array([300, 450, 620, 750, 820, 950, 1050, 1200]) - 1

# For forward
push_delta_vel_list_x = [0.25, 0.25, 0.25, 0.25, 0.3, 0.25, 0.25]
push_delta_vel_list_y = [-0.7, 0.75, -1.2, 0.6, -0.7, 0.7, -0.6]
push_delta_vel_list_z = [-0.72, -0.7, -0.72, -0.72, -0.72, -0.72, -0.72]
push_interval = np.array([300, 450, 620, 750, 850, 1000, 1050, 1200]) - 1


# @torch.jit.script
def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    world_frame_vec = torch.clone(gravity_frame_vec)
    world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                             sin_yaw * gravity_frame_vec[:, 1])
    world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                             cos_yaw * gravity_frame_vec[:, 1])
    return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    gravity_frame_vec = torch.clone(world_frame_vec)
    gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                               sin_yaw * world_frame_vec[:, 1])
    gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                               cos_yaw * world_frame_vec[:, 1])
    return gravity_frame_vec


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    # print(f"self.sim_device_id: {sim_device_id}")
    # print(f"self.graphics_device_id: {graphics_device_id}")
    # print(f"self.physics_engine: {sim_conf.physics_engine}")
    # print(f"self.sim_params: {sim_conf.sim_params}")
    sim = gym.create_sim(sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params)

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
    else:
        viewer = None

    return gym, sim, viewer


class Go2TrotEnv:

    def __init__(self,
                 num_envs: int,
                 config: ml_collections.ConfigDict(),
                 device: str = "cuda",
                 show_gui: bool = False,
                 use_real_robot: bool = False):
        self._step_cnt = 0
        self._push_cnt = 0
        self._push_magnitude = 1

        self._num_envs = num_envs
        self._device = device
        self._show_gui = show_gui
        self._config = config
        # print(f"self._config: {self._config}")
        # time.sleep(123)
        self._use_real_robot = use_real_robot

        with self._config.unlocked():
            self._config.goal_lb = to_torch(self._config.goal_lb, device=self._device)
            self._config.goal_ub = to_torch(self._config.goal_ub, device=self._device)
            if self._config.get('observation_noise', None) is not None:
                self._config.observation_noise = to_torch(
                    self._config.observation_noise, device=self._device)

        teacher_config = DictConfig(
            {"chi": 0.15, "tau": 100, "teacher_enable": True, "teacher_learn": True, "epsilon": 1,
             "cvxpy_solver": "solver"}
        )
        self.ha_teacher = HATeacher(num_envs=self._num_envs, teacher_cfg=teacher_config, device=self._device)
        self.coordinator = Coordinator(num_envs=self._num_envs, device=self._device)

        # Set up robot and controller
        use_gpu = ("cuda" in device)
        self._sim_conf = sim_config.get_config(
            use_gpu=use_gpu,
            show_gui=show_gui,
            use_penetrating_contact=self._config.get('use_penetrating_contact', False)
        )

        # Assign the desired state
        self.desired_vx = 0.7
        self.desired_com_height = 0.3
        self.desired_wz = 0.

        self._gym, self._sim, self._viewer = create_sim(self._sim_conf)
        self._create_terrain()

        # add_ground(self._gym, self._sim)
        # add_terrain(self._gym, self._sim, "stair")
        # add_terrain(self._gym, self._sim, "slope")
        # add_terrain(self._gym, self._sim, "stair", 3.95, True)
        # add_terrain(self._gym, self._sim, "stair", 0., True)

        self._indicator_flag = False
        self._indicator_cnt = 0

        self._init_positions = self._compute_init_positions()
        if self._use_real_robot:
            robot_class = go2_robot.Go2Robot
        else:
            robot_class = go2.Go2

        # The Robot Env
        self._robot = robot_class(
            num_envs=self._num_envs,
            init_positions=self._init_positions,
            sim=self._sim,
            viewer=self._viewer,
            world_env=WildTerrainEnv,
            sim_config=self._sim_conf,
            motor_control_mode=MotorControlMode.HYBRID,
            motor_torque_delay_steps=self._config.get('motor_torque_delay_steps', 0)
        )

        # self._init_buffer()

        strength_ratios = self._config.get('motor_strength_ratios', 0.7)
        if isinstance(strength_ratios, Sequence) and len(strength_ratios) == 2:
            ratios = torch_rand_float(lower=to_torch([strength_ratios[0]], device=self._device),
                                      upper=to_torch([strength_ratios[1]], device=self._device),
                                      shape=(self._num_envs, 3),
                                      device=self._device)
            # Use the same ratio for all ab/ad motors, all hip motors, all knee motors
            ratios = torch.concatenate((ratios, ratios, ratios, ratios), dim=1)
            self._robot.motor_group.strength_ratios = ratios
        else:
            self._robot.motor_group.strength_ratios = strength_ratios

        # Need to set frictions twice to make it work on GPU... 😂
        self._robot.set_foot_frictions(0.01)
        self._robot.set_foot_frictions(self._config.get('foot_friction', 1.))

        # 获取所有 actor 的名称
        actor_count = self._gym.get_actor_count(self._robot._envs[0])
        actor_names = [self._gym.get_actor_name(self._robot._envs[0], i) for i in range(actor_count)]

        # 打印所有 actor 的名称
        for name in actor_names:
            print(name)
        # 获取 root state tensor
        root_state_tensor = self._gym.acquire_actor_root_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)

        def get_gait_config():
            config = ml_collections.ConfigDict()
            config.stepping_frequency = 2  # 1
            config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
            config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            # config.initial_offset = np.array([0.15, 0.15, -0.35, -0.35]) * (2 * np.pi)
            # config.swing_ratio = np.array([0.6, 0.6, 0.6, 0.6])
            return config

        self._config.gait = get_gait_config()
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(self._robot, self._config.gait)
        self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            desired_base_height=self.desired_com_height,
            foot_height=self._config.get('swing_foot_height', 0.12),
            foot_landing_clearance=self._config.get('swing_foot_landing_clearance', 0.02)
        )
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            self._robot,
            base_position_kp=self._config.get('base_position_kp', np.array([0., 0., 50.])),
            base_position_kd=self._config.get('base_position_kd', np.array([10., 10., 10.])),
            base_orientation_kp=self._config.get('base_orientation_kp', np.array([50., 50., 0.])),
            base_orientation_kd=self._config.get('base_orientation_kd', np.array([10., 10., 10.])),
            weight_ddq=self._config.get('qp_weight_ddq', np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
            foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
            clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
            # body_inertia=self._config.get('qp_body_inertia', np.diag([0.14, 0.35, 0.35]) * 0.5),
            use_full_qp=self._config.get('use_full_qp', False)
        )

        # Set reference trajectory
        self._torque_optimizer.set_controller_reference(desired_height=self.desired_com_height,
                                                        desired_lin_vel=[self.desired_vx, 0, 0],
                                                        desired_rpy=[0., 0., 0.],
                                                        desired_ang_vel=[0., 0., self.desired_wz])
        # Path Planner
        self._planner = PathPlanner(self._robot)
        self._shortest_path = []

        self._steps_count = torch.zeros(self._num_envs, device=self._device)
        self._init_yaw = torch.zeros(self._num_envs, device=self._device)
        self._episode_length = self._config.episode_length_s / self._config.env_dt
        self._construct_observation_and_action_space()

        self._obs_buf = torch.zeros((self._num_envs, 12), device=self._device)
        self._privileged_obs_buf = None
        self._desired_landing_position = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._cycle_count = torch.zeros(self._num_envs, device=self._device)

        self._extras = dict()

        # Planning
        self._planning_flag = True
        self._save_raw = deque()
        self._step_cnt_solo = 0

        # Running a few steps with dummy commands to ensure JIT compilation
        if self._num_envs == 1 and self._use_real_robot:
            for state in range(16):
                desired_contact_state = torch.tensor(
                    [[(state & (1 << i)) != 0 for i in range(4)]], dtype=torch.bool, device=self._device)
                for _ in range(3):
                    self._gait_generator.update()
                    self._swing_leg_controller.update()
                    desired_foot_positions = self._swing_leg_controller.desired_foot_positions
                    self._torque_optimizer.get_action(
                        desired_contact_state, swing_foot_position=desired_foot_positions)

    def _init_buffer(self):
        self._robot._init_buffers()
        self._robot._post_physics_step()

    def _create_terrain(self):
        """Creates terrains.

        Note that we set the friction coefficient to all 0 here. This is because
        Isaac seems to pick the larger friction out of a contact pair as the
        actual friction coefficient. We will set the corresponding friction coefficient
        in robot friction.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = .2
        plane_params.dynamic_friction = .2
        plane_params.restitution = 0.
        self._gym.add_ground(self._sim, plane_params)
        self._terrain = None

    def _compute_init_positions(self):
        init_positions = torch.zeros((self._num_envs, 3), device=self._device)
        num_cols = int(np.sqrt(self._num_envs))
        distance = 1.
        for idx in range(self._num_envs):
            init_positions[idx, 0] = idx // num_cols * distance
            init_positions[idx, 1] = idx % num_cols * distance
            # init_positions[idx, 2] = 0.268
            init_positions[idx, 2] = 0.3
        return to_torch(init_positions, device=self._device)

    def _construct_observation_and_action_space(self):
        # robot_lb = to_torch(
        #     [0., -3.14, -3.14, -4., -4., -10., -3.14, -3.14, -3.14] +
        #     [-0.5, -0.5, -0.4] * 4,
        #     device=self._device)
        # robot_ub = to_torch([0.6, 3.14, 3.14, 4., 4., 10., 3.14, 3.14, 3.14] +
        #                     [0.5, 0.5, 0.] * 4,
        #                     device=self._device)

        robot_lb = to_torch(
            [-0.1, -0.1, 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14], device=self._device)
        robot_ub = to_torch([0.1, 0.1, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14],
                            device=self._device)

        task_lb = to_torch([-2., -2., -1., -1., -1.], device=self._device)
        task_ub = to_torch([2., 2., 1., 1., 1.], device=self._device)
        # self._observation_lb = torch.concatenate((task_lb, robot_lb))
        # self._observation_ub = torch.concatenate((task_ub, robot_ub))
        self._observation_lb = robot_lb
        self._observation_ub = robot_ub
        if self._config.get("observe_heights", False):
            num_heightpoints = len(self._config.measured_points_x) * len(
                self._config.measured_points_y)
            self._observation_lb = torch.concatenate(
                (self._observation_lb, torch.zeros(num_heightpoints, device=self._device) - 3))
            self._observation_ub = torch.concatenate(
                (self._observation_ub, torch.zeros(num_heightpoints, device=self._device) + 3))
        self._action_lb = to_torch(self._config.action_lb, device=self._device) * 5
        self._action_ub = to_torch(self._config.action_ub, device=self._device) * 5

    def reset(self) -> torch.Tensor:
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids) -> torch.Tensor:
        # Aggregate rewards
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            self._extras["episode"] = {}
            print(f"self._gait_generator.true_phase[env_ids]: {self._gait_generator.true_phase[env_ids]}")
            # time.sleep(123)
            self._extras["episode"]["cycle_count"] = torch.mean(
                self._gait_generator.true_phase[env_ids]) / (2 * torch.pi)

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self._get_privileged_observations()

            self._steps_count[env_ids] = 0
            self._cycle_count[env_ids] = 0
            self._init_yaw[env_ids] = self._robot.base_orientation_rpy[env_ids, 2]
            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)

        return self._obs_buf, self._privileged_obs_buf

    def step(self, drl_action: torch.Tensor):

        # time.sleep(1)
        # print(f"action is: {action}")
        self._last_obs_buf = torch.clone(self._obs_buf)

        self._last_action = torch.clone(drl_action)
        print(f"drl_action before clip is: {drl_action}")
        print(f"self._action_lb: {self._action_lb}")
        print(f"self._action_ub: {self._action_ub}")
        drl_action = torch.clip(drl_action, self._action_lb, self._action_ub)
        print(f"step action is: {drl_action}")
        # time.sleep(1)
        # action = torch.zeros_like(action)
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
        self._steps_count += 1
        logs = []

        start = time.time()
        # self._config.env_dt = 0.002

        for step in range(max(int(self._config.env_dt / self._robot.control_timestep), 1)):
            # print(f"config.env_dt: {self._config.env_dt}")
            # print(f"self._robot.control_timestep: {self._robot.control_timestep}")
            self._gait_generator.update()
            self._swing_leg_controller.update()

            # self._robot.state_estimator.update_ground_normal_vec()
            # self._robot.state_estimator.update_foot_contact(self._gait_generator.desired_contact_state)

            if self._planning_flag:
                # goal = [51, 2]  # In (x, y) from world frame
                goal = [49, -1]  # In (x, y) from world frame
                # self._draw_goals(goal=goal)

                # Plot the depth camera origin in world frame
                # map_origin_in_world = self._robot.camera_sensor[0].get_depth_origin_world_frame()
                # self._draw_goals(goal=map_origin_in_world)

                map_goal = self._planner.world_to_map_frame(pose_in_world=goal)

                # Human readable map goal
                # x
                # ^
                # |
                # | --> y
                print(f"map_goal: {map_goal}")

                if self._step_cnt_solo == 0:
                    self._occupancy_map = self._robot.camera_sensor[0].get_bev_map(as_occupancy=True,
                                                                                   show_map=False,
                                                                                   save_map=True)

                    self._costmap, self._costmap_for_plot = self._planner.get_costmap(goal_in_map=map_goal,
                                                                                      show_map=False)

                    # 进行高斯模糊，sigma 控制模糊程度（推荐 1~5 之间）
                    from scipy.ndimage import gaussian_filter

                    # sigma = 5
                    # self._costmap = gaussian_filter(self._costmap, sigma=sigma)
                    # self._costmap_for_plot = gaussian_filter(self._costmap_for_plot, sigma=sigma)

                    # start_pt = (200, 350)
                    curr_pos_w = np.asarray(self._robot.base_position[0, :2])
                    start_pt = self._planner.world_to_map_frame(pose_in_world=curr_pos_w)
                    start_pt = (int(start_pt[0]), int(start_pt[1]))
                    print(f"start_pt: {start_pt}")

                    # time.sleep(123)

                    def path_plot(start, goal):
                        path = self._planner.get_shortest_path(distance_map=self._costmap, start_pt=start,
                                                               goal_pt=goal)
                        # path = path[:1]

                        for i in range(len(path)):
                            # print(f"current_path: {path[i]}")
                            path_in_world = self._planner.map_to_world_frame(path[i])
                            path_in_world = [path_in_world[0], path_in_world[1]]
                            self._shortest_path.append(path_in_world)

                        # print(f"pts: {self._shortest_path}")
                        # time.sleep(123)

                        import matplotlib.pyplot as plt

                        # Visualize the costmap
                        plt.figure(figsize=(8, 8))
                        # plt.imshow(self._costmap_for_plot, cmap="coolwarm", origin="lower")
                        plt.imshow(self._costmap_for_plot, cmap='viridis', origin="lower")
                        plt.colorbar(label="Distance from Goal (m)")
                        plt.scatter(goal[1], goal[0], color="green", label="Goal", marker="x", s=100)
                        plt.scatter(start[1], start[0], color="blue", label="Start", marker="o", s=100)

                        # Visualize the shortest path
                        # path = np.array(path)
                        plt.plot(np.asarray(path)[:, 1], np.asarray(path)[:, 0], color="black", linewidth=2,
                                 label="Shortest Path")
                        plt.legend()
                        plt.title("Shortest Path on Distance Map")
                        plt.show()

                    path_plot(start=start_pt, goal=map_goal)  # Plot the shortest path

                    # Plot the shortest path
                    # self._draw_path(pts=np.asarray(self._shortest_path))

                if len(self._save_raw) == 0:
                    # Generate trajectory (yaw_rate)

                    # Position and Velocity in Map coordinates
                    # curr_pos_m = self.m2mm(self.w2m(curr_pos_w))
                    curr_pos_m = map_goal

                    ref_pos, ref_vel = self._planner.generate_ref_trajectory(map_goal=map_goal,
                                                                             occupancy_map=self._occupancy_map)
                    print(f"ref_pos: {ref_pos}")
                    print(f"ref_vel: {ref_vel}")
                    # time.sleep(123)
                    self._save_raw.extend(ref_vel)
                if len(self._save_raw) > 0:

                    # Reference speed
                    ut = self._save_raw.popleft()
                    # vel_x = ut[0] * 0.02
                    # yaw_rate = ut[1] * 18
                    yaw_rate = ut[1] * 1
                    self.desired_wz = yaw_rate

                    # Setup controller reference
                    self._torque_optimizer.set_controller_reference(
                        desired_height=self.desired_com_height,
                        # desired_lin_vel=[vel_x, 0, 0],
                        desired_lin_vel=[self.desired_vx, 0, 0],
                        desired_rpy=[0, 0, 0],
                        desired_ang_vel=[0, 0, self.desired_wz]
                    )
                # else:
                #     self._torque_optimizer.set_controller_reference(
                #         desired_height=self.desired_com_height,
                #         # desired_lin_vel=[ref_vel[5, 0], 0, 0],
                #         desired_lin_vel=[self.desired_vx, 0, 0],
                #         desired_rpy=[0, 0, 0],
                #         desired_ang_vel=[0, 0, 0]
                #     )

            # self._planner.get_costmap(goal_in_map=map_goal, show_map=True)
            # self._planner.set_goal(map_goal)

            if self._use_real_robot:
                self._robot.state_estimator.update_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error
                self._robot.update_desired_foot_contact(
                    self._gait_generator.desired_contact_state)  # pytype: disable=attribute-error

            # time.sleep(123)

            # Get swing leg action
            desired_foot_positions = self._swing_leg_controller.desired_foot_positions

            motor_action, self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.get_action(
                self._gait_generator.desired_contact_state,
                swing_foot_position=desired_foot_positions,
                generated_acc=None)

            # HP-Student action (residual form)
            # hp_action = self._desired_acc
            hp_action = drl_action + self._desired_acc

            # HA-Teacher update
            self._robot.energy_2d = self.ha_teacher.update(self._torque_optimizer.tracking_error)

            # HA-Teacher action
            ha_action, dwell_flag = self.ha_teacher.get_action()
            # ha_action = to_torch(ha_action, device=self._device)

            # Use Normal Kp Kd
            # ha_action = self._desired_acc.squeeze()

            print(f"hp_action: {hp_action}")
            print(f"ha_action: {ha_action}")
            print(f"self._torque_optimizer.tracking_error: {self._torque_optimizer.tracking_error}")
            terminal_stance_ddq, action_mode = self.coordinator.get_terminal_action(hp_action=hp_action,
                                                                                    ha_action=ha_action,
                                                                                    plant_state=self._torque_optimizer.tracking_error,
                                                                                    dwell_flag=dwell_flag,
                                                                                    epsilon=self.ha_teacher.epsilon)

            terminal_stance_ddq = to_torch(terminal_stance_ddq, device=self._device)
            print(f"terminal_stance_ddq: {terminal_stance_ddq}")
            print(f"action_mode: {action_mode}")

            # Action mode indices
            hp_indices = torch.argwhere(action_mode == ActionMode.STUDENT.value).squeeze(-1)
            ha_indices = torch.argwhere(action_mode == ActionMode.TEACHER.value).squeeze(-1)
            hp_motor_action = None
            ha_motor_action = None

            print(f"hp_indices: {hp_indices}")
            print(f"ha_indices: {ha_indices}")

            # HP-Student in Control
            if len(hp_indices) > 0:
                hp_motor_action, self._desired_acc[hp_indices], self._solved_acc[hp_indices], \
                    self._qp_cost[hp_indices], self._num_clips[hp_indices] = self._torque_optimizer.get_action(
                    self._gait_generator.desired_contact_state[hp_indices],
                    swing_foot_position=desired_foot_positions[hp_indices],
                    generated_acc=terminal_stance_ddq[hp_indices]
                )

            # HA-Teacher in Control
            if len(ha_indices) > 0:
                ha_motor_action, self._desired_acc[ha_indices], self._solved_acc[ha_indices], \
                    self._qp_cost[ha_indices], self._num_clips[ha_indices] = self._torque_optimizer.get_safe_action(
                    self._gait_generator.desired_contact_state[ha_indices],
                    swing_foot_position=desired_foot_positions[ha_indices],
                    safe_acc=terminal_stance_ddq[ha_indices]
                )

            # Unknown Action Mode
            if len(hp_indices) == 0 and len(ha_indices) == 0:
                raise RuntimeError(f"Unrecognized Action Mode: {action_mode}")

            # Add both HP-Student and HA-Teacher Motor Action
            motor_action = concatenate_motor_actions(command1=hp_motor_action, indices1=hp_indices,
                                                     command2=ha_motor_action, indices2=ha_indices)

            # print(f"desired_acc: {self._desired_acc}")
            # print(f"solved_acc: {self._solved_acc}")
            # print(f"motor_action: {motor_action}")
            # print(f"self._robot.base_angular_velocity_world_frame: {self._robot.base_angular_velocity_world_frame}")
            logs.append(
                dict(timestamp=self._robot.time_since_reset,
                     base_position=torch.clone(self._robot.base_position),
                     base_orientation_rpy=torch.clone(
                         self._robot.base_orientation_rpy),
                     base_velocity=torch.clone(self._robot.base_velocity_body_frame),
                     base_angular_velocity=torch.clone(
                         self._robot.base_angular_velocity_body_frame),
                     motor_positions=torch.clone(self._robot.motor_positions),
                     motor_velocities=torch.clone(self._robot.motor_velocities),
                     motor_action=motor_action,
                     motor_torques=self._robot.motor_torques,
                     num_clips=self._num_clips,
                     foot_contact_state=self._gait_generator.desired_contact_state,
                     foot_contact_force=self._robot.foot_contact_forces,
                     desired_swing_foot_position=desired_foot_positions,
                     desired_acc_body_frame=self._desired_acc,
                     desired_vx=self.desired_vx,
                     desired_wz=self.desired_wz,
                     desired_com_height=self.desired_com_height,
                     ha_action=ha_action,
                     hp_action=hp_action,
                     action_mode=action_mode,
                     acc_min=to_torch([-10, -10, -10, -20, -20, -20], device=self._device),
                     acc_max=to_torch([10, 10, 10, 20, 20, 20], device=self._device),
                     energy=to_torch(self._robot.energy_2d, device=self._device),
                     solved_acc_body_frame=self._solved_acc,
                     foot_positions_in_base_frame=self._robot.foot_positions_in_base_frame,
                     env_action=drl_action,
                     env_obs=torch.clone(self._obs_buf)
                     )
            )

            if self._use_real_robot:
                logs[-1]["base_acc"] = np.array(
                    self._robot.raw_state.imu.accelerometer)  # pytype: disable=attribute-error

            # Error in last step
            err_prev = self._torque_optimizer.tracking_error

            ######### Step The Motor Action #########
            self._robot.step(motor_action)
            #########################################

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self.get_privileged_observations()
            # rewards = self.get_reward()

            # Error in next step
            err_next = self._torque_optimizer.tracking_error

            # Get Lyapunov-like reward
            rewards = self.get_lyapunov_reward(err=err_prev, err_next=err_next)

            dones = torch.logical_or(dones, self._is_done())
            # print(f"rewards: {rewards.shape}")
            # print(f"dones: {dones.shape}")
            # print(f"sum_reward: {sum_reward.shape}")
            sum_reward += rewards * torch.logical_not(dones)

            # torch.manual_seed(42)
            # random_interval = torch.randint(100, 200, (1,), dtype=torch.int64, device=self.device)
            # if self._step_cnt % random_interval == 0:

            if self._indicator_flag:
                if self._indicator_cnt < 30:
                    self._indicator_cnt += 1
                else:
                    self._gym.clear_lines(self._viewer)
                    self._indicator_cnt = 0
                    self._indicator_flag = False

            # Origin push method (static interval)
            # if self._step_cnt > 298 and (self._step_cnt + 1) % 150 == 0:
            #     print(f"cnt is: {self._step_cnt}, pushing the robot now")
            #
            #     _push_robots()
            #     self._indicator_flag = True
            #     # time.sleep(1)
            #
            #     # self._gym.add_lines(self.robot._envs[0], None, 0, [])

            # New push method (Dynamic method)
            if self._step_cnt > 298 and self._step_cnt == push_interval[self._push_cnt]:
                print(f"cnt is: {self._step_cnt}, pushing the robot now")

                # _push_robots()
                self._indicator_flag = True
                # time.sleep(1)

                # self._gym.add_lines(self.robot._envs[0], None, 0, [])

            # print(f"Time: {self._robot.time_since_reset}")
            # print(f"Gait: {gait_action}")
            # print(f"Foot: {foot_action}")
            # print(f"Phase: {self._obs_buf[:, 3]}")
            # print(
            #     f"Desired Velocity: {self._torque_optimizer.desired_linear_velocity}")
            # print(f"Current Velocity: {self._robot.base_velocity_world_frame}")
            # print(
            #     f"Desired RPY: {self._torque_optimizer.desired_base_orientation_rpy}")
            # print(f"Current RPY: {self._robot.base_orientation_rpy}")
            # print(
            #     f"Desired Angular Vel: {self._torque_optimizer.desired_angular_velocity}"
            # )
            # print(
            #     f"Current Angular vel: {self._robot.base_angular_velocity_body_frame}")
            # print(f"Desired Acc: {self._desired_acc}")
            # print(f"Solved Acc: {self._solved_acc}")
            # ans = input("Any Key...")
            # if ans in ["Y", "y"]:
            #   import pdb
            #   pdb.set_trace()
            self._extras["logs"] = logs

            # Resample commands
            new_cycle_count = (self._gait_generator.true_phase / (2 * torch.pi)).long()
            finished_cycle = new_cycle_count > self._cycle_count
            env_ids_to_resample = finished_cycle.nonzero(as_tuple=False).flatten()
            self._cycle_count = new_cycle_count

            is_terminal = torch.logical_or(finished_cycle, dones)
            # if is_terminal.any():
            #     print(f"terminal_reward is: {self.get_terminal_reward(is_terminal, dones)}")

            # import pdb
            # pdb.set_trace()
            # self._resample_command(env_ids_to_resample)
            if not self._use_real_robot:
                print(f"dones: {dones}")
                self.reset_idx(dones.nonzero(as_tuple=False).flatten())
                pass

            # if dones.any():
            #   import pdb
            #   pdb.set_trace()
            # print(f"sum_reward: {sum_reward}")
            # print(f"sum_reward: {sum_reward.shape}")
            if self._show_gui:
                self._robot.render()

        self._step_cnt_solo += 1

        end = time.time()
        print(
            f"*************************************** step duration: {end - start} ***************************************")
        return self._obs_buf, self._privileged_obs_buf, sum_reward, dones, self._extras

    def _get_observations(self):

        s_desired_next = torch.cat((
            self._torque_optimizer.desired_base_position,
            self._torque_optimizer.desired_base_orientation_rpy,
            self._torque_optimizer.desired_linear_velocity,
            self._torque_optimizer.desired_angular_velocity),
            dim=-1)

        s_old_next = torch.cat((
            self._robot.base_position,
            self._robot.base_orientation_rpy,
            self._robot.base_velocity_body_frame,
            self._robot.base_angular_velocity_body_frame,
        ), dim=-1)

        robot_obs = (s_old_next - s_desired_next)

        robot_obs = self._torque_optimizer.tracking_error
        # print(f"s_desired_next: {s_desired_next}")
        # print(f"s_old_next: {s_old_next}")
        # print(f"robot_obs: {robot_obs}")
        # print(f"robot_obs: {robot_obs.shape}")

        obs = robot_obs
        if self._config.get("observation_noise", None) is not None and (not self._use_real_robot):
            obs += torch.randn_like(obs) * self._config.observation_noise
        return obs

    def get_observations(self):
        return self._obs_buf

    def _get_privileged_observations(self):
        return None

    def get_privileged_observations(self):
        return self._privileged_obs_buf

    def get_lyapunov_reward(self, err, err_next):
        """Get lyapunov-like reward
            error: position_error     (p)
                   orientation_error  (rpy)
                   linear_vel_error   (v)
                   angular_vel_error  (w)
        """

        _MATRIX_P = torch.tensor(MATRIX_P, dtype=torch.float32, device=self._device)
        s_curr = err[:, 2:]
        s_next = err_next[:, 2:]
        # print(f"s: {s.shape}")
        # print(f"s_new: {s_new.shape}")
        # ly_reward_curr = s_new.T @ MATRIX_P @ s_new
        ST1 = torch.matmul(s_curr, _MATRIX_P)
        ly_reward_curr = torch.sum(ST1 * s_curr, dim=1, keepdim=True)

        # ly_reward_next = s_next_new.T @ MATRIX_P @ s_next_new
        ST2 = torch.matmul(s_next, _MATRIX_P)
        ly_reward_next = torch.sum(ST2 * s_next, dim=1, keepdim=True)

        sum_reward = ly_reward_curr - ly_reward_next  # multiply scaler to decrease
        # print(f"sum_reward: {sum_reward.shape}")
        # print(f"sum_reward: {sum_reward}")
        # sum_reward = torch.tensor(reward, device=self._device)

        return sum_reward.squeeze(dim=-1)

    # def _episode_terminated(self):
    #     timeout = (self._steps_count >= self._episode_length)
    #     cycles_finished = (self._gait_generator.true_phase /
    #                        (2 * torch.pi)) > self._config.get('max_jumps', 1)
    #     return torch.logical_or(timeout, cycles_finished)

    def _episode_terminated(self):
        timeout = (self._steps_count >= self._episode_length)
        return timeout

    def _is_done(self):
        is_unsafe = torch.logical_or(
            self._robot.projected_gravity[:, 2] < 0.5,
            self._robot.base_position[:, 2] < self._config.get('terminate_on_height', 0.15))
        if torch.any(is_unsafe):
            print(f"self._robot.projected_gravity[:, 2]: {self._robot.projected_gravity[:, 2]}")
            print(f" self._robot.base_position[:, 2]: {self._robot.base_position[:, 2]}")
            # time.sleep(123)
        if self._config.get('terminate_on_body_contact', False):
            is_unsafe = torch.logical_or(is_unsafe, self._robot.has_body_contact)
            if torch.any(is_unsafe):
                print(f"self._robot.has_body_contact: {self._robot.has_body_contact}")
                # time.sleep(123)

        if self._config.get('terminate_on_limb_contact', False):
            limb_contact = torch.logical_or(self._robot.calf_contacts, self._robot.thigh_contacts)
            limb_contact = torch.sum(limb_contact, dim=1)
            is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)
            if torch.any(is_unsafe):
                print(f"limb_contact: {limb_contact}")
                # time.sleep(123)

        # print(self._robot.base_position[:, 2])
        # input("Any Key...")
        # if is_unsafe.any():
        #   import pdb
        #   pdb.set_trace()
        return torch.logical_or(self._episode_terminated(), is_unsafe)

    @property
    def device(self):
        return self._device

    @property
    def robot(self):
        return self._robot

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def desired_landing_position(self):
        return self._desired_landing_position

    @property
    def action_space(self):
        return self._action_lb, self._action_ub

    @property
    def observation_space(self):
        return self._observation_lb, self._observation_ub

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_obs(self):
        return self._observation_lb.shape[0]

    @property
    def num_privileged_obs(self):
        return None

    @property
    def num_actions(self):
        return self._action_lb.shape[0]

    @property
    def max_episode_length(self):
        return self._episode_length

    @property
    def episode_length_buf(self):
        return self._steps_count

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._steps_count = to_torch(new_length, device=self._device)
        self._gait_generator._current_phase += 2 * torch.pi * (new_length / self.max_episode_length * self._config.get(
            'max_jumps', 1) + 1)[:, None]
        self._cycle_count = (self._gait_generator.true_phase /
                             (2 * torch.pi)).long()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """

        push_rng = np.random.default_rng(seed=42)
        # time.sleep(2)
        # self.arrow_plot()

        # max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_vel = 1
        # torch.manual_seed(30)
        # self.robot._root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
        #                                                    device=self.device)  # lin vel x/y

        curr_vx = self.robot._root_states[:self._num_envs, 7]
        curr_vy = self.robot._root_states[:self._num_envs, 8]
        curr_vz = self.robot._root_states[:self._num_envs, 9]
        print(f"self.robot._root_states: {self.robot._root_states}")
        # time.sleep(123)

        sgn = -1 if dual_push_sequence[self._push_cnt] % 2 == 0 else 1
        # if self._push_cnt == 0:
        #     self._push_magnitude *= sgn
        # elif sgn == -1:
        #     self._push_magnitude = -1.1
        # elif sgn == 1:
        #     self._push_magnitude = 1.9
        # self._push_magnitude = push_magnitude_list[self._push_cnt]
        # Original static push
        # delta_x = -0.25
        # delta_y = 0.62 * self._push_magnitude
        # delta_z = -0.72

        delta_x = push_delta_vel_list_x[self._push_cnt]
        delta_y = push_delta_vel_list_y[self._push_cnt]
        delta_z = push_delta_vel_list_z[self._push_cnt]

        # Random push from uniform distribution
        # new_seed = push_sequence[self._push_cnt]
        # print(f"new_seed: {new_seed}")
        # np.random.seed(new_seed)
        # sgn = -1 if new_seed % 2 == 0 else 1
        #
        # delta_x = np.random.uniform(low=0., high=0.2) * sgn
        # delta_y = np.random.uniform(low=0., high=0.6) * sgn
        # delta_z = np.random.uniform(low=-0.8, high=-0.7)
        print(f"delta_x: {delta_x}")
        print(f"delta_y: {delta_y}")
        print(f"delta_z: {delta_z}")
        # time.sleep(2)

        vel_after_push_x = curr_vx + delta_x
        vel_after_push_y = curr_vy + delta_y
        vel_after_push_z = curr_vz + delta_z

        print(f"vel_after_push_x: {vel_after_push_x}")
        print(f"vel_after_push_y: {vel_after_push_y}")
        print(f"vel_after_push_z: {vel_after_push_z}")

        # Turn on the push indicator for viewing
        self.draw_push_indicator(target_pos=[delta_x, delta_y, delta_z])

        # time.sleep(1)
        # print(f"delta_x: {curr_x}")
        self.robot._root_states[:self._num_envs, 7] = torch.full((self.num_envs, 1), vel_after_push_x.item())
        self.robot._root_states[:self._num_envs, 8] = torch.full((self.num_envs, 1), vel_after_push_y.item())
        # print(f"FFFFFFFFFFFFFFFFFFFFFFFFFFFF: {torch.full((self.num_envs, 1), .2)}")
        self.robot._root_states[:self._num_envs, 9] = torch.full((self.num_envs, 1), vel_after_push_z.item())
        # self._gym.set_actor_root_state_tensor(self._sim, gymtorch.unwrap_tensor(self.robot._root_states))

        actor_count = self._gym.get_env_count(self._sim)
        # self.robot._root_states = self.robot._root_states.repeat(7, 1)
        indices = to_torch([i for i in range(self._num_envs)], dtype=torch.int32, device=self._device)
        indices_tensor = gymtorch.unwrap_tensor(indices)
        self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(self.robot._root_states),
                                                      indices_tensor,
                                                      1)
        self._push_cnt += 1
        # self._gym.set_dof_state_tensor(self._sim, gymtorch.unwrap_tensor(self.robot._root_states))

    def draw_push_indicator(self, target_pos=[1., 0., 0.]):
        """Draw the line indicator for pushing the robot"""

        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.01, 50, 50, None, color=(1, 0., 0.))
        pose_robot = self.robot._root_states[:self._num_envs, :3].squeeze(dim=0).cpu().numpy()
        print(f"pose_robot: {pose_robot}")
        self.target_pos_rel = to_torch([target_pos], device=self._device)
        for i in range(5):
            norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
            target_vec_norm = self.target_pos_rel / (norm + 1e-5)
            print(f"norm: {norm}")
            print(f"target_vec_norm: {target_vec_norm}")
            # pose_arrow = pose_robot[:3] + 0.1 * (i + 3) * target_vec_norm[:self._num_envs, :3].cpu().numpy()

            xy = pose_robot[:2] + 0.08 * (i + 3) * target_vec_norm[:self._num_envs, :2].cpu().numpy()
            z = pose_robot[2] + 0.03 * (i + 3) * target_vec_norm[:self._num_envs, 2].cpu().numpy()
            print(f"xy: {xy}")
            print(f"xy: {z}")
            pose_arrow = np.hstack((xy.squeeze(), z))
            # pose_arrow = pose_arrow.squeeze()
            print(f"pose_arrow: {pose_arrow}")
            pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_arrow[2]), r=None)
            print(f"pose: {pose}")
            gymutil.draw_lines(sphere_geom_arrow, self._gym, self._viewer, self.robot._envs[0], pose)

        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
        # for i in range(5):
        #     norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        #     target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        #     pose_arrow = pose_robot[:2] + 0.2 * (i + 3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
        #     pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
        #     gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _draw_goals(self, goal, env_ids=0):
        # Red
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))

        # Blue
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))

        # Green
        sphere_geom_reached = gymutil.WireframeSphereGeometry(0.2, 32, 32, None, color=(0, 1, 0))
        sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
        sphere_geom_arrow2 = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))

        pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], 0), r=None)
        gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)
        # pose = gymapi.Transform(gymapi.Vec3(goal[0] + 1, goal[1] + 1, 0), r=None)
        # gymutil.draw_lines(sphere_geom_cur, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)
        # pose = gymapi.Transform(gymapi.Vec3(goal[0] + 2, goal[1] + 2, 0), r=None)
        # gymutil.draw_lines(sphere_geom_reached, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)
        # pose = gymapi.Transform(gymapi.Vec3(goal[0] + 3, goal[1] + 3, 0), r=None)
        # gymutil.draw_lines(sphere_geom_arrow, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)
        # pose = gymapi.Transform(gymapi.Vec3(goal[0] + 4, goal[1] + 4, 0), r=None)
        # gymutil.draw_lines(sphere_geom_arrow2, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)

    def _draw_path(self, pts, env_ids=0):
        # Red
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 32, 32, None, color=(0, 0, 0))
        print(f"pts: {pts}")
        print(f"pts: {pts.shape}")
        for i in range(pts.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(pts[i, 0], pts[i, 1], 0), r=None)
            gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._robot.env_handles[env_ids], pose)
