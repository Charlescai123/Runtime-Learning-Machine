import os
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras import Model
import tensorflow_probability as tfp

tfd = tfp.distributions

@dataclass
class SACConfig:
    agent_name: str = 'SAC'
    mode: str = 'train'
    soft_alpha: float = 0.005
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    batch_size: int = 128
    target_network_frequency: int = 1
    noise_clip: float = 0.5
    entropy_alpha: float = 0.1
    gamma_discount: float = 0.99
    model_path: str = ''
    total_training_steps: int = 500000
    replay_buffer_size: int = 500000
    learning_starts: int = 3000
    policy_update_frequency: int = 1
    use_layer_norm: bool = False

    # Network architecture configurations
    actor_hidden_units: list = field(default_factory=lambda: [256, 256])
    critic_hidden_units: list = field(default_factory=lambda: [256, 256])
    actor_activation: str = 'relu'
    critic_activation: str = 'relu'


LOG_STD_MAX = 2
LOG_STD_MIN = -5


def build_mlp(input_layer, hidden_units, activation, use_layer_norm=False):
    """Builds an MLP given a starting input layer."""
    x = input_layer
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)
        if use_layer_norm:
            x = LayerNormalization()(x)
    return x


def build_actor(shape_input, shape_output, config: SACConfig, name=''):
    input_layer = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float32)
    # Build hidden layers
    x = build_mlp(input_layer, config.actor_hidden_units, config.actor_activation, use_layer_norm=config.use_layer_norm)
    mean = Dense(shape_output, activation=None, name=name + 'mean')(x)
    log_std = Dense(shape_output, activation='tanh', name=name + 'std')(x)
    # rescale log_std to [LOG_STD_MIN, LOG_STD_MAX]
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    model = Model(inputs=input_layer, outputs=[mean, log_std])
    return model


def build_critic(shape_input, shape_output, config: SACConfig, name=''):
    input_layer = Input(shape=(shape_input,), name=name + 'input', dtype=tf.float32)
    x = build_mlp(input_layer, config.critic_hidden_units, config.critic_activation,
                  use_layer_norm=config.use_layer_norm)
    q_value = Dense(shape_output, activation=None, name=name + 'q_value')(x)
    model = Model(inputs=input_layer, outputs=q_value)
    return model


@tf.function
def get_action_logp(mean, log_std):
    std = tf.math.exp(log_std)
    normal = tfd.Normal(mean, std)
    eps = tfd.Normal(loc=tf.zeros_like(std), scale=tf.ones_like(std)).sample()
    x_t = mean + eps * std
    action = tf.math.tanh(x_t)
    log_prob = normal.log_prob(x_t)
    # log probability correction for tanh squashing
    log_prob -= tf.math.log((1 - tf.math.pow(action, 2)) + 1e-5)
    log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
    return action, log_prob


class SACAgent:
    def __init__(self, params: SACConfig, shape_observations, shape_action, mode='train'):
        self.params = params
        self.shape_action = shape_action
        self.critic_update_step = 0

        # Build models
        self.actor = build_actor(shape_observations, shape_action, params, name="actor_")
        self.critic = build_critic(shape_observations + shape_action, 1, params, name="critic_")
        self.critic_target = build_critic(shape_observations + shape_action, 1, params, name="critic_target_")
        self.critic_2 = build_critic(shape_observations + shape_action, 1, params, name="critic_2_")
        self.critic_2_target = build_critic(shape_observations + shape_action, 1, params, name="critic_2_target_")

        self.value_fcn = build_critic(shape_observations, 1, params, name="value_fcn")

        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_critic_2 = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)
        self.optimizer_value_fcn = tf.keras.optimizers.Adam(learning_rate=self.params.learning_rate_critic)

        if self.params.model_path != '':
            self.load_weights(self.params.model_path, mode=mode)
        else:
            self.hard_update()

        self.get_summary()

    def get_summary(self):
        self.actor.summary()
        self.critic.summary()

    def save_weights(self, model_save_path):
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.actor.save_weights(os.path.join(model_save_path, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(model_save_path, "critic.weights.h5"))
        self.critic_target.save_weights(os.path.join(model_save_path, "critic_target.weights.h5"))
        self.critic_2.save_weights(os.path.join(model_save_path, "critic_2.weights.h5"))
        self.critic_2_target.save_weights(os.path.join(model_save_path, "critic_2_target.weights.h5"))
        print(f'Model weights are saved to {model_save_path}')

    def load_weights(self, model_path, mode='train'):
        self.actor.load_weights(os.path.join(model_path, "actor.weights.h5"))
        if mode == "train":
            self.critic.load_weights(os.path.join(model_path, "critic.weights.h5"))
            self.critic_target.load_weights(os.path.join(model_path, "critic_target.weights.h5"))
            self.critic_2.load_weights(os.path.join(model_path, "critic_2.weights.h5"))
            self.critic_2_target.load_weights(os.path.join(model_path, "critic_2_target.weights.h5"))
        print(f'Model weights are loaded from {model_path}')

    def hard_update(self):
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

    @tf.function
    def soft_update(self):
        soft_alpha = tf.convert_to_tensor(self.params.soft_alpha, dtype=tf.float32)
        self._soft_update(soft_alpha)

    @tf.function
    def _soft_update(self, soft_alpha):
        for w_new, w_old in zip(self.critic.weights, self.critic_target.weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))
        for w_new, w_old in zip(self.critic_2.weights, self.critic_2_target.weights):
            w_old.assign(w_new * soft_alpha + w_old * (1. - soft_alpha))

    def get_action(self, observations, mode='train'):
        if len(observations.shape) < 2:
            observations = tf.expand_dims(observations, 0)
        mean, log_std = self.actor(observations)
        if mode == 'train':
            action, _ = get_action_logp(mean, log_std)
        elif mode == 'test':
            # Deterministic action for evaluation
            action = tf.math.tanh(mean)
        else:
            raise NotImplementedError(f"{mode} is not implemented")
        return action.numpy()

    def optimize_joint(self, RL_buffer, context_buffer, ib_model, batch_size):
        """ Joint optimization of SAC and IB objective.

        Args:
            RL_buffer: Replay buffer for RL
            context_buffer: Replay buffer for IB
            ib_model: IB model
            batch_size: batch size for optimization

        """

        # 1) sample from RL buffer
        s, a, r, s_next, ter, c_key = RL_buffer.sample(batch_size)
        s_tf = tf.convert_to_tensor(s, dtype=tf.float32)
        a_tf = tf.convert_to_tensor(a, dtype=tf.float32)
        r_tf = tf.convert_to_tensor(r, dtype=tf.float32)
        s_next_tf = tf.convert_to_tensor(s_next, dtype=tf.float32)
        ter_tf = tf.convert_to_tensor(ter, dtype=tf.float32)

        r_tf = tf.expand_dims(r_tf, axis=-1)
        ter_tf = tf.expand_dims(ter_tf, axis=-1)
        if len(a_tf.shape) == 1:
            a_tf = tf.expand_dims(a_tf, axis=-1)

        # 2) sample IB encoder input paired with c
        ib_input = context_buffer.sample_by_context(batch_size, c_key)

        # 3) Z ~ IB encoder
        z_tf = ib_model.sample_z(ib_input, use_target=False, return_type=tf.Tensor)

        # 4) concatenate z with s and s_next
        s_aug = tf.concat([s_tf, z_tf], axis=-1)
        s_next_aug = tf.concat([s_next_tf, z_tf],
                               axis=-1)  # TODO: try resample z. Would it make the critic more robust?

        # 5) optimize SAC with augmented state
        critic_loss, critic_loss_2, q_e, q_e_2, _ = self._optimize_critic(s_aug, a_tf, r_tf, s_next_aug, ter_tf)
        self.critic_update_step += 1
        self.soft_update()

        # 6) optimize actor
        if self.critic_update_step % self.params.policy_update_frequency == 0:
            actor_loss, log_pi = self._optimize_actor(s_aug)
        else:
            actor_loss = 0.0
            log_pi = 0.0

        training_info = {
            "critic_loss": critic_loss.numpy().mean(),
            "critic_loss_2": critic_loss_2.numpy().mean(),
            "q_value": q_e.numpy().mean(),
            "q_value_2": q_e_2.numpy().mean(),
            "actor_loss": np.array(actor_loss).mean(),
            "log_pi": np.array(log_pi).mean(),
        }

        return training_info

    @tf.function
    def _optimize_critic(self, ob1, a1, r1, ob2, ter):
        mean_next, log_std_next = self.actor(ob2)
        a_next, log_prob_next = get_action_logp(mean_next, log_std_next)
        critic_target_input = tf.concat([ob2, a_next], axis=-1)
        critic_input = tf.concat([ob1, a1], axis=-1)

        q_e = self.critic_target(critic_target_input)
        q_e_2 = self.critic_2_target(critic_target_input)
        min_q_e = tf.minimum(q_e, q_e_2) - self.params.entropy_alpha * log_prob_next
        y_exp = r1 + self.params.gamma_discount * min_q_e * (1 - ter)

        with tf.GradientTape() as tape:
            y_pre = self.critic(critic_input)
            critic_loss = tf.keras.losses.MeanSquaredError()(y_exp, y_pre)
        q_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

        with tf.GradientTape() as tape2:
            y_pre_2 = self.critic_2(critic_input)
            critic_loss_2 = tf.keras.losses.MeanSquaredError()(y_exp, y_pre_2)
        q_grads_2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)

        with tf.GradientTape() as tape3:
            state_value = self.value_fcn(ob1)
            value_loss = tf.keras.losses.MeanSquaredError()(state_value, min_q_e)
        value_grades = tape3.gradient(value_loss, self.value_fcn.trainable_variables)

        self.optimizer_critic.apply_gradients(zip(q_grads, self.critic.trainable_variables))
        self.optimizer_critic_2.apply_gradients(zip(q_grads_2, self.critic_2.trainable_variables))
        self.optimizer_value_fcn.apply_gradients(zip(value_grades, self.value_fcn.trainable_variables))

        return critic_loss, critic_loss_2, q_e, q_e_2, value_loss

    @tf.function
    def _optimize_actor(self, ob1):
        with tf.GradientTape() as tape:
            mean, log_std = self.actor(ob1)
            action, log_pi = get_action_logp(mean, log_std)
            critic_input = tf.concat([ob1, action], axis=-1)
            min_q_value = tf.minimum(self.critic(critic_input), self.critic_2(critic_input))
            actor_loss = tf.reduce_mean(self.params.entropy_alpha * log_pi - min_q_value)
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        return actor_loss, log_pi

    def optimize(self, mini_batch):
        """ Baseline optimization of SAC."""
        for i in range(len(mini_batch)):
            if len(mini_batch[i].shape) == 1:
                mini_batch[i] = np.expand_dims(mini_batch[i], axis=1)

        ob1_tf = tf.convert_to_tensor(mini_batch[0], dtype=tf.float32)
        a1_tf = tf.convert_to_tensor(mini_batch[1], dtype=tf.float32)
        r1_tf = tf.convert_to_tensor(mini_batch[2], dtype=tf.float32)
        ob2_tf = tf.convert_to_tensor(mini_batch[3], dtype=tf.float32)
        ter_tf = tf.convert_to_tensor(mini_batch[4], dtype=tf.float32)

        critic_loss, critic_loss_2, q_e, q_e_2, value_loss = self._optimize_critic(ob1_tf, a1_tf, r1_tf, ob2_tf, ter_tf)
        self.critic_update_step += 1
        self.soft_update()

        if self.critic_update_step % self.params.policy_update_frequency == 0:
            actor_loss, log_pi = self._optimize_actor(ob1_tf)
        else:
            actor_loss = 0.0
            log_pi = 0.0

        training_info = {
            "critic_loss": critic_loss.numpy().mean(),
            "critic_loss_2": critic_loss_2.numpy().mean(),
            "q_value": q_e.numpy().mean(),
            "q_value_2": q_e_2.numpy().mean(),
            "actor_loss": np.array(actor_loss).mean(),
            "log_pi": np.array(log_pi).mean(),
            "value_loss": np.array(value_loss).mean(),
        }

        return training_info

