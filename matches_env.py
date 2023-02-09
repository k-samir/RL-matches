import pygame
import gym
import numpy as np
from tqdm import tqdm
from gym.utils.env_checker import check_env
import matplotlib.pyplot as plt


class MatchesEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, n_matches=20):
        self.n_matches = n_matches
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.MultiDiscrete(np.array([19])) #   self.n_matches + 1)
        self.reset()
        
    def reset(self):
        self.matches_left = self.n_matches
        self.player = 1
        return self.matches_left
    
    def step(self, action, doPrint):
        action+=1
        if action >= 1 and action <= 3:
            self.matches_left -= action
        else:
            raise Exception("Invalid Action") 

        done = False
        if self.matches_left <= 0:
            done = True
            reward = 1000 if self.player == 1 else -1000
        else:
            self.player = (self.player + 1) % 2
            reward = -1

        if doPrint:
          print("-- Player ", self.player, " --")
          print("Action ", action)
      
        return self.matches_left, reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            matches_display = '|' * self.matches_left
            print("Matches : " + matches_display)
            print("Left : ", self.matches_left)
    
    def random_step(self, doPrint):
        action = np.random.randint(3)
        return self.step(action, doPrint)

def plot_stats(stats,smooth=10):
    rows = len(stats)
    cols = 1

    fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

    for i, key in enumerate(stats):
        vals = stats[key]
        vals = [np.mean(vals[i-smooth:i+smooth]) for i in range(smooth, len(vals)-smooth)]
        if len(stats) > 1:
            ax[i].plot(range(len(vals)), vals)
            ax[i].set_title(key, size=18)
        else:
            ax.plot(range(len(vals)), vals)
            ax.set_title(key, size=18)
    plt.tight_layout()
    plt.show()  

def playRandomGame():
    env = MatchesEnv()
    done = False
    while not done:
        if env.player == 1:
            action = np.random.randint(3)
            observation, reward, done, _ = env.step(action, True)
            env.render()
        else:
            observation, reward, done, _ = env.random_step(True)
            env.render()
    print("Player " + str(env.player) + " lost.")