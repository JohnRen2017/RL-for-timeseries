import os
import sys

sys.path.append(r"C:\PythonCode\myRL\RL4TimeSeries\")
# sys.path.append(r"..\Render")
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ACmodel import Actor, Critic
from utils import Memory, Utils
from environment import Environ


class Agent:
    def __init__(
        self,
        action_dim,
        filters_C,
        kernel_size,
        hidden_R,
        dropout,
        dropout_r,
        Hstep,
        activation,
        is_training_mode,
    ):
        self.policy_clip = 0.2
        self.value_clip = 0.2
        self.entropy_coef = 0.0
        self.vf_loss_coef = 0.5
        self.minibatch = 32
        self.PPO_epochs = 10

        # TODO use predicted results
        action_std = 1.0

        self.cov_mat = tf.fill([action_dim], action_std ** 2)
        self.is_training_mode = is_training_mode

        self.actor = Actor(
            action_dim,
            filters_C,
            kernel_size,
            hidden_R,
            dropout,
            dropout_r,
            activation,
            Hstep,
        )
        self.actor_old = Actor(
            action_dim,
            filters_C,
            kernel_size,
            hidden_R,
            dropout,
            dropout_r,
            activation,
            Hstep,
        )

        self.critic = Critic(
            filters_C, kernel_size, hidden_R, dropout, dropout_r, activation, Hstep
        )
        self.critic_old = Critic(
            filters_C, kernel_size, hidden_R, dropout, dropout_r, activation, Hstep
        )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
        self.memory = Memory()
        self.utils = Utils()

    def save_eps(self, state, reward, action, done, next_state):
        self.memory.save_eps(state, reward, action, done, next_state)

    def get_loss(self, states, actions, rewards, dones, next_states):
        action_mean, values = self.actor(states), self.critic(states)
        old_action_mean, old_values = self.actor_old(states), self.critic_old(states)
        next_values = self.critic(next_states)

        Old_values = tf.stop_gradient(old_values)

        # getting external general advantages estimator
        Advantages = tf.stop_gradient(
            self.utils.generalized_advantage_estimation(
                values, rewards, next_values, dones
            )
        )
        Returns = tf.stop_gradient(
            self.utils.temporal_difference(rewards, next_values, dones)
        )

        # find the ratio (pi_theta / pi_theta_old)
        logprobs = tf.expand_dims(
            self.utils.logprob(action_mean, self.cov_mat, actions), 1
        )
        Old_logprobs = tf.expand_dims(
            self.utils.logprob(old_action_mean, self.cov_mat, actions), 1
        )

        # getting entropy from the action probability
        dist_entropy = tf.math.reduce_mean(
            self.utils.entropy(action_mean, self.cov_mat)
        )

        # getting external critic loss by using clipped critic value
        vpredclipped = old_values + tf.clip_by_value(
            values - Old_values, -self.value_clip, self.value_clip
        )  # minimize the difference between old value and new value
        vf_losses1 = tf.math.square(Returns - values)
        vf_losses2 = tf.math.square(Returns - vpredclipped)
        critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2)) * 0.0

        # finding surrogate loss
        ratios = tf.math.exp(logprobs - Old_logprobs)
        surr1 = ratios * Advantages
        surr2 = (
            tf.clip_by_value(ratios, 1 - self.policy_clip, 1 + self.policy_clip)
            * Advantages
        )
        pg_loss = tf.math.reduce_mean(tf.math.minimum(surr1, surr2))

        # need to maximize policy loss to make agent always find better rewards
        # and minimize critic loss
        loss = (
            (critic_loss * self.vf_loss_coef)
            - (dist_entropy * self.entropy_coef)
            - pg_loss
        )
        return loss

    @tf.function
    def act(self, state):
        state = tf.expand_dims(tf.cast(state, dtype=tf.float32), 0)
        action_mean = self.actor(state)

        # don't need to sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # sample the action
            action = self.utils.sample(action_mean, self.cov_mat)
            action = tf.clip_by_value(action, clip_value_min=0.0, clip_value_max=1.0)
        else:
            action = action_mean
        return tf.squeeze(action)

    # get loss and do backpropagation for PPO part (the actor and critic)
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            loss = self.get_loss(states, actions, rewards, dones, next_states)

        gradients = tape.gradient(
            loss, self.actor.trainable_variables + self.critic.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.actor.trainable_variables + self.critic.trainable_variables,
            )
        )

    # update the model
    def update_ppo(self):
        batch_size = int(self.memory.length() / self.minibatch)

        # optimize policy for K epochs
        for _ in range(self.PPO_epochs):
            for (
                states,
                actions,
                rewards,
                dones,
                next_states,
            ) in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(states, actions, rewards, dones, next_states)

        # clear the memory
        self.memory.clearMemory()

        # copy new weights into old policy
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_weights(self):
        if not os.path.exists("weights"):
            os.makedirs("weights")
        self.actor.save_weights(r"weights/actor.hd5")
        self.actor_old.save_weights(r"weights/actor_old.hd5")
        self.critic.save_weights(r"weights/critic.hd5")
        self.critic_old.save_weights(r"weights/critic_old.hd5")

    def load_weights(self):
        self.actor.load_weights(r"weights/actor.hd5")
        self.actor_old.load_weights(r"weights/actor_old.hd5")
        self.critic.load_weights(r"weights/critic.hd5")
        self.critic_old.load_weights(r"weights/critic_old.hd5")


def plot(datas):
    print("----------")

    plt.plot(datas)
    plt.plot()
    plt.xlabel("Episode")
    plt.ylabel("Datas")
    plt.show()

    print("Max :", np.max(datas))
    print("Min :", np.min(datas))
    print("Avg :", np.mean(datas))


def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update):
    state = env.reset()
    done = False
    total_reward = 0
    eps_time = 0
    ############################################

    while not done:
        action = [agent.act(state).numpy()]
        action_gym = 2 * np.array(action)

        next_state, reward, done = env.step(action_gym)

        eps_time += 1
        t_updates += 1
        total_reward += reward

        if training_mode:
            agent.save_eps(state, reward, action, done, next_state)

        state = next_state

        if render:
            env.render()

        if training_mode:
            if t_updates % n_update == 0:
                agent.update_ppo()
                t_updates = 0

        if done:
            return total_reward, eps_time, t_updates


def main():
    ############## Hyperparameters ##############
    # using_google_drive = (
    #     False
    # )  # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = True  # If you want to load the agent, set this to True
    save_weights = False  # If you want to save the agent, set this to True
    training_mode = (
        False
    )  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = (
        300
    )  # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

    render = (
        True
    )  # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 500  # How many episode before you update the Policy
    n_plot_batch = 1000  # How many episode you want to plot the result
    n_episode = 5000  # How many episode you want to run
    #############################################
    env = Environ(
        data=None, max_iteration=24 * 30, lookback=24 * 7, genlength=10_000_000
    )

    state_dim = (24 * 7, 3)
    action_dim = 3

    filters_C = 32
    kernel_size = 5
    hidden_R = 36
    dropout = 0.5
    dropout_r = 0.5
    activation = "relu"
    Hstep = 12

    agent = Agent(
        action_dim,
        filters_C,
        kernel_size,
        hidden_R,
        dropout,
        dropout_r,
        Hstep,
        activation,
        training_mode,
    )
    #############################################

    # if using_google_drive:
    #     from google.colab import drive

    #     drive.mount("/test")

    if load_weights:
        agent.load_weights()
        print("Weight Loaded")

    rewards = []
    batch_rewards = []
    batch_solved_reward = []

    times = []
    batch_times = []

    t_updates = 0

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates = run_episode(
            env, agent, state_dim, render, training_mode, t_updates, n_update
        )
        print(
            "Episode {} \t t_reward: {} \t time: {} \t ".format(
                i_episode, int(total_reward), time
            )
        )
        batch_rewards.append(int(total_reward))
        batch_times.append(time)

        if save_weights:
            agent.save_weights()
            print("weights saved")

        if reward_threshold:
            if len(batch_solved_reward) == 100:
                if np.mean(batch_solved_reward) >= reward_threshold:
                    for reward in batch_rewards:
                        rewards.append(reward)

                    for time in batch_times:
                        times.append(time)

                    print("You solved task after {} episode".format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)

        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)

            for reward in batch_rewards:
                rewards.append(reward)

            for time in batch_times:
                times.append(time)

            batch_rewards = []
            batch_times = []

            print("========== Cummulative ==========")
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)

    print("========== Final ==========")
    # Plot the reward, times for every episode
    plot(rewards)
    plot(times)


if __name__ == "__main__":
    main()
