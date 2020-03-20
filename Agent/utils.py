import tensorflow as tf
import tensorflow_probability as tfp


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def save_eps(self, state, reward, action, done, next_state):
        self.rewards.append(reward)
        self.states.append(state.tolist())
        self.actions.append(action)
        self.dones.append(float(done))
        self.next_states.append(next_state.tolist())

    def clearMemory(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()

    def length(self):
        return len(self.actions)

    def get_all_items(self):
        states = tf.constant(self.states, dtype=tf.float32)
        actions = tf.constant(self.actions, dtype=tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype=tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype=tf.float32), 1)
        next_states = tf.expand_dims(tf.constant(self.next_states, dtype=tf.float32), 1)
        return tf.data.Dataset.from_tensor_slices(
            (states, actions, rewards, dones, next_states)
        )


class Utils:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.0

    # if you want to create agent for continuous action environment, you must find the proper
    # distribution for it (some people use Multivariate Gaussian Distribution), and making the
    # neural network output directly the action, not probability (Deterministic policy)

    def sample(self, mean, cov_mat):
        distribution = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)
        return distribution.sample()

    def entropy(self, mean, cov_mat):
        distribution = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)
        return distribution.entropy()

    def logprob(self, mean, cov_mat, value_data):
        distribution = tfp.distributions.MultivariateNormalDiag(mean, cov_mat)
        return distribution.logprob(value_data)

    def normalize(self, data):
        data_normalized = (data - tf.math.reduce_mean(data)) / (
            tf.math.reduce_std(data) + 1e-8
        )
        return data_normalized

    def temporal_difference(self, rewards, next_values, dones):
        # find TD values, TD = R + V(St+1)
        TD = rewards + self.gamma * next_values * (1 - dones)
        return TD

    def generalized_advantage_estimation(self, values, rewards, next_values, done):
        # computing general advantages estimator
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * next_values[step] * (1 - done[step])
                - values[step]
            )
            gae = delta + (self.lam * gae)
            returns.insert(0, gae)

        return tf.stack(returns)
