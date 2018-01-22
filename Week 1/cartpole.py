import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.signal import savgol_filter
import time


# a simple demo for the work of cart pole
def demo():
    # Initialize environment
    env = gym.make("CartPole-v0")
    env.seed(0)
    # number of episodes
    NUM_OF_EPISODES = 10
    results = []
    for _ in range(NUM_OF_EPISODES):
        count = 0
        env.reset()
        observation, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        while not done:
            if observation[2] > 0:
                action = 1
            else:
                action = 0
            observation, reward, done, _ = env.step(action)
            env.render()
            # print(observation)
            count += 1
            # time.sleep(0.25)
        results.append(count)
    print(np.mean(results))
    plt.plot(results)
    plt.show()
    env.close()


# implementation for Q-learning solution

class Qlearning:
    def __init__(self, env,
                 discount_rate=0.99,
                 num_of_max_steps=350):

        self.learning_rate = 0
        self.discount_rate = discount_rate
        self.exploration_rate = 0
        self.env = env
        self.num_of_max_steps = num_of_max_steps
        self.state = None
        self.action = None
        self.MIN_EXPLORE_RATE = 0.01
        self.MIN_LEARNING_RATE = 0.1
        self.state_bound = list(zip(self.env.observation_space.low, self.env.observation_space.high))

        # set max steps for the environment
        env._max_episode_steps = num_of_max_steps
        # create number of bins
        self._state_bins = [1, 1, 6, 3]
        # Create a clean Q-Table.
        self._num_actions = 2

        self.state_bound[1] = [-0.5, 0.5]
        self.state_bound[3] = [-math.radians(50), math.radians(50)]

        self.q = np.zeros(shape=(self._state_bins + [self._num_actions,]))

    def train(self, num_of_episodes=490, visualize_utils_episodes=9999):
        result_track = np.zeros(num_of_episodes)
        for episode in range(num_of_episodes):

            # init environment
            observation = self.env.reset()
            action = self.init_episode(observation)

            for step in range(num_of_episodes):
                # Visualize if allowed
                if episode > visualize_utils_episodes:
                    self.env.render()
                # Perform the action and observe the new state.
                observation, reward, done, _ = self.env.step(action)

                # Update the display and log the current state.

                # If the episode has ended prematurely, penalize the agent.
                if done and step < self.num_of_max_steps - 1:
                    result_track[episode] = step + 1
                    reward = -num_of_episodes

                # Get the next action from the learner, given our new state.
                action = self.take_action(observation, reward)

                # Record this episode to the history and check if the goal has been reached.
                if done or step == self.num_of_max_steps - 1:
                    result_track[episode] = step + 1
                    if episode >visualize_utils_episodes:
                        print("Episode {} number of steps :{}".format(episode, step))
                    break

            self.update_parameter(episode)
        return result_track

    def init_episode(self, observation):

        # Get the action for the initial state.
        self.state = self.build_state(observation)
        return np.argmax(self.q[self.state])

    def take_action(self, observation, reward):
        next_state = self.build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        if random.random() < self.exploration_rate:
            next_action = self.env.action_space.sample()
        # Select the action with the highest q
        else:
            next_action = np.argmax(self.q[next_state])

        self.q[self.state + (self.action,)] += self.learning_rate * \
                                           (reward + self.discount_rate * np.amax(self.q[next_state]) - self.q[
                                               self.state + (self.action,)])

        self.state = next_state
        self.action = next_action
        return next_action

    @staticmethod
    def digitize_value(value, bins):
        return np.digitize(x=value, bins=bins)

    def build_state(self, observation):
        state = []
        for i in range(len(observation)):
            if observation[i] <= self.state_bound[i][0]:
                bucket_index = 0
            elif observation[i] >= self.state_bound[i][1]:
                bucket_index = self._state_bins[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = self.state_bound[i][1] - self.state_bound[i][0]
                offset = (self._state_bins[i] - 1) * self.state_bound[i][0] / bound_width
                scaling = (self._state_bins[i] - 1) / bound_width
                bucket_index = int(round(scaling * observation[i] - offset))
            state.append(bucket_index)
        return tuple(state)

    def update_parameter(self, step):
        self.learning_rate = max(self.MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((step + 1) / 25)))
        self.exploration_rate = max(self.MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((step + 1) / 25)))


def main():
    # demo()
    env = gym.make("CartPole-v0")
    env.seed(0)
    random.seed(0)
    ql = Qlearning(env, num_of_max_steps=1000)
    result_set = ql.train(num_of_episodes=500,visualize_utils_episodes=250)
    smooth = savgol_filter(result_set, 101, 3)
    plt.plot(result_set)
    plt.plot(smooth,linewidth =2)

    plt.legend(['raw', 'smooth line'])
    plt.xlabel('number of episodes')
    plt.ylabel('max steps')
    plt.show()


if __name__ == '__main__':
    main()