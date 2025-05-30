import math
import os
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from runtime.student.utils import OrnsteinUhlenbeckActionNoise
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

@dataclass
class DDPGConfig:
    agent_name: str = 'DDPG'
    mode: str = 'train'
    action_noise: str = "OU"
    action_noise_factor: float = 1
    action_noise_half_decay_time: float = 1e6
    soft_alpha: float = 0.005
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    batch_size: int = 128
    gamma_discount: float = 0.99
    model_path: str = ''
    total_training_steps: int = 800000
    replay_buffer_size: int = 800000
    learning_starts: int = 1000  # this value should be larger than the batch size
    agent_vs_context_training_ratio: int = 5
    use_ground_truth_context: bool = True
    iterations_per_step: int = 1
    max_episode_steps: int = 1000
    eval_period: int = 5
    add_actions_observations: bool = False
    n_action_step: int = 5 # the dimension of action observation, should be action_shape * length


def build_mlp_model(shape_input, shape_output, name='', output_activation=None):
    input = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float16)
    dense1 = Dense(128, activation='relu', name=name + 'dense1')(input)
    dense2 = Dense(128, activation='relu', name=name + 'dense2')(dense1)
    dense3 = Dense(128, activation='relu', name=name + 'dense3')(dense2)
    output = Dense(shape_output, activation=output_activation, name=name + 'output')(dense3)
    model = Model(inputs=input, outputs=output, name=name)
    return model

class DDPGAgent:
    def __init__(self, params: DDPGConfig, shape_observations, shape_action, mode='train'):
        self.params = params
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.exploration_steps = 0  # for exploration action noise decay
        self.action_shape = shape_action
        self.observation_shape = shape_observations

        if self.params.model_path != '':
            print(f"Model path: {self.params.model_path}")
            if self.actor is None:
                self.create_model(shape_observations, shape_action)
            self.load_weights(self.params.model_path, mode=mode)
            print("Pretrained model loaded...")
        else:
            self.create_model(shape_observations, shape_action)
            self.hard_update()

        self.add_action_noise = True
        if self.params.action_noise == 'no':
            self.add_action_noise = False
        elif self.params.action_noise == 'OU':
            self.action_noise = OrnsteinUhlenbeckActionNoise(shape_action)
        else:
            raise NotImplementedError(f"{self.params.action_noise} noise is not implemented")

    def save_weights(self, model_save_path):

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        self.actor.save_weights(os.path.join(model_save_path, "actor.weights.h5"))
        self.actor_target.save_weights(os.path.join(model_save_path, "actor_target.weights.h5"))
        self.critic.save_weights(os.path.join(model_save_path, "critic.weights.h5"))
        self.critic_target.save_weights(os.path.join(model_save_path, "critic_target.weights.h5"))

    def load_weights(self, model_path, mode='train'):

        self.actor.load_weights(os.path.join(model_path, "actor.weights.h5"))

        if mode == "train":
            self.actor_target.load_weights(os.path.join(model_path, "actor_target.weights.h5"))
            self.critic.load_weights(os.path.join(model_path, "critic.weights.h5"))
            self.critic_target.load_weights(os.path.join(model_path, "critic_target.weights.h5"))

        print("Pretrained weights are loaded")

    def create_model(self, shape_observations, shape_action):

        self.actor = build_mlp_model(shape_observations, shape_action, name="actor", output_activation='tanh')
        self.actor_target = \
            build_mlp_model(shape_observations, shape_action, name="actor_target", output_activation='tanh')
        self.critic = build_mlp_model(shape_observations + shape_action, 1, name="critic")
        self.critic_target = build_mlp_model(shape_observations + shape_action, 1, name="critic_target")
        self.actor.summary()
        self.critic.summary()

    def hard_update(self):  # double check the initialization
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

    def soft_update(self):
        soft_alpha = tf.convert_to_tensor(self.params.soft_alpha, dtype=tf.float32)
        self._soft_update(soft_alpha)

    @tf.function
    def _soft_update(self, soft_alpha):
        # Obtain weights directly as tf.Variables
        actor_weights = self.actor.weights
        actor_target_weights = self.actor_target.weights
        critic_weights = self.critic.weights
        critic_target_weights = self.critic_target.weights

        for w_new, w_old in zip(actor_weights, actor_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

        for w_new, w_old in zip(critic_weights, critic_target_weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

    def get_action(self, observations, mode='train'):
        if mode == 'train':
            return self.get_exploration_action(observations)
        else:
            return self.get_exploitation_action(observations)

    def get_exploration_action(self, observations):
        observations_tensor = tf.expand_dims(observations, 0)
        action = tf.squeeze(self.actor(observations_tensor)).numpy()  # squeeze to kill batch_size
        if  self.add_action_noise:
            action_noise = self.action_noise.sample() * self.params.action_noise_factor
            action = np.clip((action + action_noise), a_min=-1, a_max=1, dtype=float)
            self.exploration_steps += 1
            self.noise_factor_decay()
        return action

    def get_exploitation_action(self, observations):
        observations_tensor = tf.expand_dims(observations, 0)
        action_exploitation = self.actor(observations_tensor)
        return tf.squeeze(action_exploitation).numpy()

    def noise_factor_decay(self):
        decay_rate = 0.693 / self.params.action_noise_half_decay_time
        self.params.action_noise_factor *= math.exp(-decay_rate * self.exploration_steps)

    def optimize(self, mini_batch):
        if self.optimizer_critic is None:
            self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        if self.optimizer_actor is None:
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)

        ob1_tf = tf.convert_to_tensor(mini_batch[0], dtype=tf.float32)
        a1_tf = tf.convert_to_tensor(mini_batch[1], dtype=tf.float32)
        r1_tf = tf.convert_to_tensor(mini_batch[2], dtype=tf.float32)
        ob2_tf = tf.convert_to_tensor(mini_batch[3], dtype=tf.float32)
        ter_tf = tf.convert_to_tensor(mini_batch[4], dtype=tf.float32)

        critic_loss, q_value = self._optimize_critic(ob1_tf, a1_tf, r1_tf, ob2_tf, ter_tf)
        actor_loss = self._optimize_actor(ob1_tf)
        self.soft_update()

        training_info = {"critic_loss": critic_loss.numpy().mean(),
                         "q_value": q_value.numpy().mean(),
                         "actor_loss": actor_loss.numpy().mean()}

        return training_info

    @tf.function
    def _optimize_critic(self, ob1, a1, r1, ob2, ter):
        with tf.GradientTape() as tape:
            a2 = self.actor_target(ob2)
            critic_target_input = tf.concat([ob2, a2], axis=-1)
            q_e = self.critic_target(critic_target_input)  # q value
            y_exp = r1 + self.params.gamma_discount * q_e * (1 - ter)
            critic_input = tf.concat([ob1, a1], axis=-1)
            y_pre = self.critic(critic_input)
            critic_loss = tf.keras.losses.MeanSquaredError()(y_exp, y_pre)

        q_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(q_grads, self.critic.trainable_variables))
        return critic_loss, q_e

    @tf.function
    def _optimize_actor(self, ob1):
        with tf.GradientTape() as tape:
            a1_predict = self.actor(ob1)
            critic_input = tf.concat([ob1, a1_predict], axis=-1)
            actor_loss = -1 * tf.math.reduce_mean(self.critic(critic_input))
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        return actor_loss
