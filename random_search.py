import tensorflow as tf
import numpy as np
import gym
import logging
import time

class Harness:

    def run_episode(self, env, agent):
        observation = env.reset()
        print(observation.shape)
        total_reward = 0
        for _ in range(1000):
            action = agent.next_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward


class LinearAgent:

    def __init__(self):
        self.parameters = np.random.rand(4) * 2 - 1

    def next_action(self, observation):
        return 0 if np.matmul(self.parameters, observation) < 0 else 1


def random_search():
    # implement this!
    env = gym.make('CartPole-v0')
    best_params = None
    best_reward = 0
    agent =  LinearAgent()
    harness = Harness()

    for step in range(1000):
        env.render()
        time.sleep(0.1)
        agent.parameters = np.random.rand(4)*2 -1
        reward = harness.run_episode(env,agent)
        if step % 5 == 0: # for every 100 steps
            print(reward)
        if reward > best_reward:
            best_reward = reward
            best_params = agent.parameters
        if reward == 200:
                break
    return reward



reward = random_search()
print(reward)
