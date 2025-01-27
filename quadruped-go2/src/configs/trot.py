"""Configuration for Go2 Trot Env"""
from ml_collections import ConfigDict

from src.envs import go2_trot_env
import torch
import numpy as np


def get_training_config():
    """Config for training"""
    config = ConfigDict()
    config.seed = 1

    policy_config = ConfigDict()
    policy_config.init_noise_std = .5
    policy_config.actor_hidden_dims = [512, 256, 128]
    policy_config.critic_hidden_dims = [512, 256, 128]
    policy_config.activation = "elu"
    config.policy = policy_config

    alg_config = ConfigDict()
    alg_config.value_loss_coef = 1.0
    alg_config.use_clipped_value_loss = True
    alg_config.clip_param = 0.2
    alg_config.entropy_coef = 0.01
    alg_config.num_learning_epochs = 5
    alg_config.num_mini_batches = 4
    alg_config.learning_rate = 1e-3
    alg_config.schedule = "adaptive"
    alg_config.gamma = 0.99
    alg_config.lam = 0.95
    alg_config.desired_kl = 0.01
    alg_config.max_grad_norm = 1.
    config.algorithm = alg_config

    runner_config = ConfigDict()
    runner_config.policy_class_name = "ActorCritic"
    runner_config.algorithm_class_name = "DDPG"
    runner_config.num_steps_per_env = 24
    runner_config.save_interval = 50
    runner_config.experiment_name = "ddpg_trot"
    runner_config.max_iterations = 500
    config.runner = runner_config
    return config


def get_env_config():
    """Config for Environment"""

    config = ConfigDict()

    # HA-Teacher
    ha_teacher_config = ConfigDict()
    ha_teacher_config.chi = 0.15
    ha_teacher_config.tau = 100
    ha_teacher_config.enable = False
    ha_teacher_config.correct = True
    ha_teacher_config.epsilon = 1
    ha_teacher_config.cvxpy_solver = "solver"
    config.ha_teacher = ha_teacher_config

    # Gait config
    gait_config = ConfigDict()
    gait_config.stepping_frequency = 2
    gait_config.initial_offset = np.array([0., 0.5, 0.5, 0.], dtype=np.float32) * (2 * np.pi)
    gait_config.swing_ratio = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    config.gait = gait_config

    # Fully Flexible
    config.observation_lb = np.array([-0.01, -0.01, 0., -3.14, -3.14, -3.14, -1., -1., -1., -3.14, -3.14, -3.14])
    config.observation_ub = np.array([0.01, 0.01, 0.6, 3.14, 3.14, 3.14, 1., 1., 1., 3.14, 3.14, 3.14])
    config.action_lb = np.array([-10., -10., -10., -20., -20., -20.])
    config.action_ub = np.array([10., 10., 10., 20., 20., 20.])

    config.episode_length_s = 20.
    config.max_jumps = 10.
    config.env_dt = 0.01
    # config.env_dt = 0.002
    config.motor_strength_ratios = 1.
    config.motor_torque_delay_steps = 5
    config.use_yaw_feedback = False

    # Stance controller
    config.base_position_kp = np.array([0., 0., 25])
    config.base_position_kd = np.array([5., 5., 5.])
    config.base_orientation_kp = np.array([25., 25., 0.])
    config.base_orientation_kd = np.array([5., 5., 5.])
    config.qp_foot_friction_coef = 0.7
    config.qp_weight_ddq = np.diag([1., 1., 10., 10., 10., 1.])
    config.qp_body_inertia = np.array([0.14, 0.35, 0.35]) * 1.5
    config.use_full_qp = False
    config.clip_grf_in_sim = True
    config.foot_friction = 0.7  # 0.7

    # Swing controller
    config.swing_foot_height = 0.12
    config.swing_foot_landing_clearance = 0.02

    # Termination condition
    config.terminate_on_body_contact = False
    config.terminate_on_limb_contact = False
    config.terminate_on_height = 0.15
    config.use_penetrating_contact = False

    # Reward
    config.rewards = [
        ('upright', 0.02),
        ('contact_consistency', 0.008),
        ('foot_slipping', 0.032),
        ('foot_clearance', 0.008),
        ('out_of_bound_action', 0.01),
        ('knee_contact', 0.064),
        ('stepping_freq', 0.008),
        ('com_distance_to_goal_squared', 0.016),
        ('com_height', 0.01),
    ]
    config.clip_negative_reward = False
    config.normalize_reward_by_phase = True

    config.terminal_rewards = []
    config.clip_negative_terminal_reward = False
    return config


def get_config():
    """Main entrance for the parsing the config"""
    config = ConfigDict()
    config.training = get_training_config()
    config.env_class = go2_trot_env.Go2TrotEnv
    config.environment = get_env_config()
    return config
