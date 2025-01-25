import os
import time
import copy
import imageio
import warnings
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from numpy import linalg as LA
import matplotlib.pyplot as plt

from cartpole.src.physical_design import MATRIX_P, F
from cartpole.src.ha_teacher.ha_teacher import HATeacher
from cartpole.src.hp_student.agents.ddpg import DDPGAgent
from cartpole.src.hp_student.agents.replay_mem import ReplayMemory
from cartpole.src.coordinator.coordinator import Coordinator
from cartpole.src.utils.utils import ActionMode, energy_value, logger
from cartpole.src.envs.cart_pole import observations2state, state2observations
from cartpole.src.envs.cart_pole import Cartpole, get_init_condition
from cartpole.src.logger.logger import Logger, plot_trajectory
from cartpole.src.utils.utils import check_dir

np.set_printoptions(suppress=True)


class Trainer:
    def __init__(self, config):
        self.params = config
        self.gamma = config.hp_student.phydrl.gamma  # Contribution ratio of Data/Model action

        # Environment (Cartpole simulates Real-Plant)
        self.cartpole = Cartpole(config.cartpole)

        # HP Student
        self.hp_params = config.hp_student
        self.agent_params = config.hp_student.agents
        self.shape_observations = self.cartpole.state_observations_dim
        self.shape_action = self.cartpole.action_dim
        self.replay_mem = ReplayMemory(config.hp_student.agents.replay_buffer.buffer_size)
        self.agent = DDPGAgent(agent_cfg=config.hp_student.agents,
                               taylor_cfg=config.hp_student.taylor,
                               shape_observations=self.shape_observations,
                               shape_action=self.shape_action,
                               mode=config.logger.mode)

        # HA Teacher
        self.ha_params = config.ha_teacher
        self.ha_teacher = HATeacher(teacher_cfg=config.ha_teacher, cartpole_cfg=config.cartpole)

        # Coordinator
        self.coordinator = Coordinator()

        # Logger and Plotter
        self.logger = Logger(config.logger)

        # Variables for caching
        self._initial_loss = self.agent_params.initial_loss
        self._teacher_learn = self.ha_teacher.teacher_learn
        self._action_magnitude = config.hp_student.agents.action.magnitude
        self._max_steps_per_episode = self.agent_params.max_steps_per_episode
        self._terminate_on_failure = self.params.cartpole.terminate_on_failure
        self._exp_prefill_size = self.agent_params.replay_buffer.experience_prefill_size
        self._batch_size = self.agent_params.replay_buffer.batch_size

        self.failed_times = 0

    def interaction_step(self, mode=None):

        current_state = copy.deepcopy(self.cartpole.state)
        observations, _ = state2observations(current_state)

        self.ha_teacher.update(state=np.asarray(current_state[:4]))  # Teacher update

        terminal_action, nominal_action = self.get_final_action(state=current_state,
                                                                mode=mode)
        # Update logs
        self.logger.update_logs(
            state=copy.deepcopy(self.cartpole.state[:4]),
            action=self.coordinator.plant_action,
            action_mode=self.coordinator.action_mode,
            energy=energy_value(state=np.array(self.cartpole.state[:4]),
                                p_mat=MATRIX_P)
        )

        # Inject Terminal Action
        next_state = self.cartpole.step(action=terminal_action)

        observations_next, failed = state2observations(next_state)

        if self.coordinator.action_mode == ActionMode.TEACHER:
            ha_flag = True
        else:
            ha_flag = False
        reward, distance_score = self.cartpole.reward_fcn(current_state, nominal_action, next_state, ha_flag=ha_flag)

        return observations, nominal_action, observations_next, failed, reward, distance_score, ha_flag

    def train(self):
        episode = 0
        global_steps = 0
        best_dsas = 0.0  # Best distance score and survived
        moving_average_dsas = 0.0
        optimize_time = 0

        # teacher_intervention_cnt = 0
        # teacher_action_cnt = 0
        # student_action_cnt = 0

        initial_num = 0
        # reward_folder_path = f"./data/iclr/unsafe_cl/reward/unsafe_cl_drl/"
        # reward_folder_path = f"./data/iclr/unsafe_cl/reward/unsafe_cl_phydrl2/"
        # reward_folder_path = f"./data/simplex_compare/reward/cl_simplex_phydrl/"
        # reward_folder_path = f"./data/iclr/learning_compare/unlearn/initial1"
        reward_folder_path = f"./data/iclr/learning_compare/learn/initial1/"
        # reward_folder_path = f"./data/iclr/dwell_time/"
        teacher_statistic_path = f"./data/iclr/teacher_statistic/initial1_1000.txt"

        check_dir(reward_folder_path)

        # Run for max training episodes
        for ep_i in range(int(self.agent_params.max_training_episodes)):

            # teacher_intervention_cnt = 0
            # teacher_action_cnt = 0
            # student_action_cnt = 0

            # cond = [-0.024478718101496655, -0.5511401911926881, 0.43607751272052686, -0.25180280548180833, False]
            # Reset all modules
            if self.params.cartpole.random_reset.train:
                self.cartpole.random_reset()
            else:
                self.cartpole.reset()

            self.ha_teacher.reset(state=np.array(self.cartpole.state[:4]))
            self.coordinator.reset(state=np.array(self.cartpole.state[:4]),
                                   epsilon=self.ha_teacher.epsilon)

            # Logging clear for each episode
            self.logger.clear_logs()

            print(f"Training at {ep_i} init_cond: {self.cartpole.state[:4]}")
            pbar = tqdm(total=self._max_steps_per_episode, desc="Episode %d" % ep_i)
            # continue

            reward_list = []
            distance_score_list = []
            critic_loss_list = []
            failed = False

            ep_steps = 0

            ep_filename = f"{reward_folder_path}/episode{ep_i}.txt"
            # ep_filename = f"{reward_folder_path}/initial3.txt"
            # file.write("\nAppending this line to the file.")
            for step in range(self._max_steps_per_episode):

                # if ep_i == 3:
                #     time.sleep(1)
                observations, action, observations_next, failed, r, distance_score, ha_flag = \
                    self.interaction_step(mode='train')

                if self.ha_params.teacher_correct is False and ha_flag is True:
                    pass
                else:
                    with open(ep_filename, 'a') as file:
                        rwd = np.squeeze(r)
                        file.write(f'{rwd}\n')
                        file.close()
                    reward_list.append(r)
                    distance_score_list.append(distance_score)
                    self.replay_mem.add((observations, action, r, observations_next, failed))

                    if self.replay_mem.size > self._exp_prefill_size:
                        minibatch = self.replay_mem.sample(self._batch_size)
                        critic_loss = self.agent.optimize(minibatch)
                        optimize_time += 1
                    else:
                        critic_loss = self._initial_loss
                    critic_loss_list.append(critic_loss)

                global_steps += 1
                ep_steps += 1
                pbar.update(1)

                if failed and self._terminate_on_failure:
                    # print(f"step is: {step}")
                    self.failed_times += 1 * failed
                    print(f"Cartpole system failed, terminate for safety concern!")
                    pbar.close()
                    break

            # Plot Phase
            if self.params.logger.fig_plotter.phase.plot:
                self.logger.plot_phase(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    epsilon=self.ha_params.epsilon,
                    p_mat=MATRIX_P,
                    idx=ep_i
                )

            # Plot Trajectories
            if self.params.logger.fig_plotter.trajectory.plot:
                self.logger.plot_trajectory(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    action_set=self.params.cartpole.force_bound,
                    freq=self.params.cartpole.frequency,
                    idx=ep_i
                )

            mean_reward = np.mean(reward_list)
            mean_distance_score = np.mean(distance_score_list)
            mean_critic_loss = np.mean(critic_loss_list)

            # Write trajectories to file
            with open(teacher_statistic_path, 'a+') as f:
                teacher_action_cnt = len([action_mode for action_mode in self.logger.action_mode_list if
                                          action_mode == ActionMode.TEACHER])
                student_action_cnt = len([action_mode for action_mode in self.logger.action_mode_list if
                                          action_mode == ActionMode.STUDENT])
                f.write(
                    f"episode {ep_i}: teacher_activation_times: {self.ha_teacher.activation_cnt}, "
                    f"teacher_action_cnt: {teacher_action_cnt}, "
                    f"student_action_cnt: {student_action_cnt}\n")
            f.close()

            self.logger.log_training_data(mean_reward, mean_distance_score, mean_critic_loss, failed, global_steps)

            print(f"Average_reward: {mean_reward:.6}\n"
                  f"Distance_score: {mean_distance_score:.6}\n"
                  f"Critic_loss: {mean_critic_loss:.6}\n"
                  f"Total_steps_ep: {ep_steps} ")

            # Save weights per episode
            self.agent.save_weights(self.logger.model_dir)

            if (ep_i + 1) % self.hp_params.agents.evaluation_period == 0:
                eval_mean_reward, eval_mean_distance_score, eval_failed = self.evaluation(mode='eval', idx=ep_i)
                self.logger.change_mode(mode='train')  # Change mode back
                self.logger.log_evaluation_data(eval_mean_reward, eval_mean_distance_score, eval_failed,
                                                global_steps)
                moving_average_dsas = 0.95 * moving_average_dsas + 0.05 * eval_mean_distance_score
                if moving_average_dsas > best_dsas:
                    self.agent.save_weights(self.logger.model_dir + '-best')
                    best_dsas = moving_average_dsas

            episode += 1
            # print(f"global_steps is: {global_steps}")

            # Whether to terminate training based on training_steps
            if global_steps > self.agent_params.max_training_steps and self.agent_params.training_by_steps:
                np.savetxt(f"{self.logger.log_dir}/failed_times.txt",
                           [self.failed_times, episode, self.failed_times / episode])
                print(f"Final_optimize time: {optimize_time}")
                print("Total failed:", self.failed_times)
                exit("Reach maximum steps, exit...")

        np.savetxt(f"{self.logger.log_dir}/failed_times.txt",
                   [self.failed_times, episode, self.failed_times / episode])
        print(f"Final_optimize time: {optimize_time}")
        print("Total failed:", self.failed_times)
        exit("Reach maximum episodes, exit...")

    def evaluation(self, reset_state=None, mode=None, idx=0):

        # trajectory_file_path = f'./data/iclr/hierarchical_learning/safe/'
        # trajectory_file_path = f'./data/iclr/hierarchical_learning/high-performance/'

        # trajectory_file_path = f'./data/iclr/unsafe_cl/trajectory/unsafe_cl_drl/ep1/trajectory1.txt'
        # trajectory_file_path = f'./data/simplex_compare/cl_simplex_drl/ep20/cl_simplex_drl_trajectory1.txt'
        # trajectory_file_path = f'./data/simplex_compare/cl_simplex_phydrl/ep20/cl_simplex_phydrl_trajectory1.txt'

        if self.params.cartpole.random_reset.eval:
            self.cartpole.random_reset()
        else:
            self.cartpole.reset(reset_state)

        print(f"Evaluating at {idx} init_cond: {self.cartpole.state[:4]}")

        reward_list = []
        distance_score_list = []
        failed = False
        trajectory_tensor = []
        ani_frames = []

        self.logger.change_mode(mode=mode)  # Change mode
        self.logger.clear_logs()  # Clear logs

        # Visualization flag
        visual_flag = (self.params.logger.live_plotter.animation.show
                       or self.params.logger.live_plotter.live_trajectory.show)

        if visual_flag:
            plt.ion()

        for step in range(self.agent_params.max_evaluation_steps):
            observations, action, observations_next, failed, r, distance_score, _ = \
                self.interaction_step(mode=mode)

            # Visualize Cart-pole animation
            if self.params.logger.live_plotter.animation.show:
                frame = self.cartpole.render(mode='rgb_array', idx=step)
                ani_frames.append(frame)

            # Visualize Live trajectory
            if self.params.logger.live_plotter.live_trajectory.show:
                self.logger.live_plotter.animation_run(
                    x_set=self.params.cartpole.safety_set.x,
                    theta_set=self.params.cartpole.safety_set.theta,
                    action_set=self.params.cartpole.force_bound,
                    state=self.logger.state_list[-1],
                    action=self.logger.action_list[-1],
                    action_mode=self.logger.action_mode_list[-1],
                    energy=self.logger.energy_list[-1],
                )
                plt.pause(0.01)
            reward_list.append(r)
            distance_score_list.append(distance_score)

            if failed and self.params.cartpole.terminate_on_failure:
                break

        # Write trajectories to file
        # trajectory_file_path += f'trajectory{idx}.txt'
        # with open(trajectory_file_path, 'w') as f:
        #     for state in self.logger.state_list:
        #         line = ' '.join(map(str, state))
        #         f.write(f"{line}\n")
        # f.close()

        # with open(trajectory_file_path, 'w') as file:
        #     np.savetxt(trajectory_file_path, np.asarray(self.logger.state_list))
        #     file.close()
        mean_reward = np.mean(reward_list)
        mean_distance_score = np.mean(distance_score_list)

        # Save as a GIF (Cart-pole animation)
        if self.params.logger.live_plotter.animation.save_to_gif:
            if len(ani_frames) == 0:
                warnings.warn("Failed to save animation as gif, please set animation.show to True")
            else:
                logger.debug(f"Animation frames: {ani_frames}")
                last_frame = ani_frames[-1]
                for _ in range(5):
                    ani_frames.append(last_frame)
                gif_path = self.params.logger.live_plotter.animation.gif_path
                fps = self.params.logger.live_plotter.animation.fps
                print(f"Saving animation frames to {gif_path}")
                imageio.mimsave(gif_path, ani_frames, fps=fps, loop=0)

        # Save as a GIF (Cart-pole trajectory)
        if self.params.logger.live_plotter.live_trajectory.save_to_gif:
            if len(ani_frames) == 0:
                warnings.warn("Failed to save live trajectory as gif, please set live_trajectory.show to True")
            else:
                last_frame = self.logger.live_plotter.frames[-1]
                for _ in range(5):
                    self.logger.live_plotter.frames.append(last_frame)
                gif_path = self.params.logger.live_plotter.live_trajectory.gif_path
                fps = self.params.logger.live_plotter.live_trajectory.fps
                print(f"Saving live trajectory frames to {gif_path}")
                imageio.mimsave(gif_path, self.logger.live_plotter.frames, fps=fps, loop=0)

        # Close and reset
        if visual_flag:
            self.cartpole.close()
            self.logger.live_plotter.reset()
            plt.ioff()
            plt.close()

        # Plot Phase
        if self.params.logger.fig_plotter.phase.plot:
            self.logger.plot_phase(
                x_set=self.params.cartpole.safety_set.x,
                theta_set=self.params.cartpole.safety_set.theta,
                epsilon=self.ha_params.epsilon,
                p_mat=MATRIX_P,
                idx=idx
            )

        # Plot Trajectory
        if self.params.logger.fig_plotter.trajectory.plot:
            self.logger.plot_trajectory(
                x_set=self.params.cartpole.safety_set.x,
                theta_set=self.params.cartpole.safety_set.theta,
                action_set=self.params.cartpole.force_bound,
                freq=self.params.cartpole.frequency,
                idx=idx
            )

        # Reset live plotter
        self.logger.live_plotter.reset()

        return mean_reward, mean_distance_score, failed

    def test2(self):
        for i in range(100):
            self.evaluation(mode='test', idx=i)

    def test(self):
        self.evaluation(mode='test', reset_state=self.params.cartpole.initial_condition)

    def get_final_action(self, state, mode=None):

        observations, _ = state2observations(state)
        s = np.asarray(state[:4])

        # DRL Action
        drl_raw_action = self.agent.get_action(observations, mode)

        # Add unknown unknowns
        if self.agent_params.unknown_distribution.apply:
            # print(f"apply unknown unknowns to the drl agent")
            drl_raw_action += self.cartpole.get_unknown_distribution()
        # print(f"drl_raw_action: {drl_raw_action}")

        drl_action = drl_raw_action * self._action_magnitude

        # Model-based Action
        phy_action = F @ s

        # Student Action (Residual form)
        hp_action = drl_action * 1 + phy_action * self.gamma

        # Teacher Action
        ha_action, dwell_flag = self.ha_teacher.get_action()
        # hp_action = ha_action

        # Terminal Action by Coordinator
        logger.debug(f"ha_action: {ha_action}")
        logger.debug(f"hp_action: {hp_action}")
        terminal_action, action_mode = self.coordinator.get_terminal_action(hp_action=hp_action, ha_action=ha_action,
                                                                            plant_state=s, dwell_flag=dwell_flag,
                                                                            epsilon=self.ha_teacher.epsilon)
        # print(f"terminal_action: {terminal_action}")
        # terminal_action, action_mode = ha_action, ActionMode.TEACHER

        # Decide nominal action to store into replay buffer
        if action_mode == ActionMode.TEACHER:
            if self._teacher_learn:  # Learn from teacher action
                nominal_action = (ha_action - phy_action) / self._action_magnitude
            else:
                nominal_action = drl_raw_action
        elif action_mode == ActionMode.STUDENT:
            nominal_action = drl_raw_action
        else:
            raise NotImplementedError(f"Unknown action mode: {action_mode}")

        return terminal_action, nominal_action
