import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy

# Initialize environment
env = gym.make("CartPole-v0")
# number of episodes
NUM_OF_EPISODES = 10
# number of bucket per parameter
N_BINS = [6, 6, 6, 6]
MIN_VALUES = [-0.5, -2.0, -0.5, -3.0]
MAX_VALUES = [0.5, 2.0, 0.5, 3.0]
BINS = [[np.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in range(4)]]


# a simple for the work of cart pole
def demo():
    results = []
    for _ in range(NUM_OF_EPISODES):
        count = 0
        env.reset()
        observation, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        while True:
            if observation[2] > 0:
                action = 1
            else:
                action = 0
            if abs(observation[0]) < 2:
                observation, reward, done, _ = env.step(action)
                env.render()
                print(observation)
                count += 1
                # time.sleep(0.25)

            else:
                results.append(count)
                break
    print(np.mean(results))
    env.close()


# implementation for q-learning solution

def get_best_result():
    return None


class Qlearning:
    def __init__(self, env, learning_rate=0.1, discount_rate=0.1, exploration_rate=0.01, episodes=100):

        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.env = env
        self.episodes = episodes
        self.cached_result = None
        self.num_of_iterations = 100
        self.q_value = {}

    def train(self, print_result=True, visualize_results=True, decay=False):

        # check if model are already trained
        if self.cached_result is not None:
            return self.cached_result

        else:
            policy = {}
            for _ in range(self.episodes):

                # reinitialize environment
                observation = self.env.reset()

                # due to problem when hashing numpy array we convert it into python tuple
                state = self.formalize(observation)

                # init first value
                self.q_value[(state, 1)] = np.random.rand()

                # iteration within maximum steps
                for _ in range(self.num_of_iterations):

                    # choose action
                    action = self.choose_action(state, self.exploration_rate)

                    # taking action
                    old_state = deepcopy(state)
                    observation, reward, done, _ = self.env.step(action)
                    state = self.formalize(observation)
                    self.env.render()
                    new_action = self.choose_action(state, self.exploration_rate)

                    # update q_value

                    self.q_value[(old_state, action)] = self.q_value[(old_state, action)] + self.learning_rate * (
                            reward + self.discount_rate * self.q_value[(state, new_action)] - self.q_value[(old_state, action)])

                    # till terminate
                    if done:
                        break

    def choose_action(self, state, exploartion_rate):
        return 1

    def formalize(self, observation):
        print(observation)
        print(BINS)
        return tuple([int(np.digitize(observation[i], BINS[i])) for i in range(4)])


def main():
    # demo()
    time.sleep(0.25)
    env = gym.make("CartPole-v0")
    ql = Qlearning(env)
    ql.train()


if __name__ == '__main__':
    main()
