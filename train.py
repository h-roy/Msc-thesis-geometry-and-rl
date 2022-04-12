import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from collections import deque
from torch.distributions.categorical import Categorical
import torch_ac
import torch.optim as optim
from torch_ac.utils import DictList
from torch.nn.functional import mse_loss
import torch.distributions as td
from env import CrossingEnv2, preprocess
from gym_minigrid.wrappers import *
from utils import ReplayBuffer
from agent import Geometry_Agent
import itertools


action_1 = [2]
action_2 = [1, 2, 0]
action_3 = [0, 2, 1]
action_4 = [1, 1, 2, 0, 0]
actions_dict = {0: action_1, 1: action_2, 2: action_3, 3: action_4}


def train_encoder(n_episodes, seed, n_episode_len, updates, batch_size, agent, env):
  for i_episode in range(n_episodes):
    env.seed(seed)
    obs = env.reset()
    obs = preprocess(obs)
    sum_reward = 0
    for t in range(n_episode_len):
        agent_coord = env.agent_pos
        agent_orient = env.agent_dir
        a = np.random.randint(0,4)
        for act in actions_dict[a]:
          next_obs, reward, done, _ = env.step(act)
        sum_reward += reward
        a2 = np.random.randint(0,4)
        for act in actions_dict[a2]:
          next_obs_2, reward, done, _ = env.step(act)
        sum_reward += reward
        next_obs = preprocess(next_obs)
        next_obs_2 = preprocess(next_obs_2)
        agent.memory.add(obs, a, reward, next_obs, done, agent_coord, agent_orient, a2, next_obs_2)
        obs = next_obs_2
        if done or t == n_episode_len-1:
            print("Episode {} finished after {} timesteps".format(i_episode+1, t+1))
            print("Reward: {}".format(sum_reward))
            rewards.append(sum_reward)
        if done:
            break
        for i in range(updates):
            loss, l_inv, l_smoothness, l_orthogonal, l_contr = agent.update_encoder(batch_size)
            print('update_step: %3d Total Loss: %5f Inverse loss: %5f Smoothness loss %5f Orthogonal loss %5f Contrastive loss %5f'
                    % (t*n_episode_len + i,
                      loss.item(),
                       l_inv.item(),
                       smootheness_coeff**2 * l_smoothness.item(),
                       smootheness_coeff * l_orthogonal.item(),
                       contrastive_coeff * l_contr.item()))



#Training Loop:
env = ImgObsWrapper(RGBImgObsWrapper(CrossingEnv2()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=512
sample = 5000
actions = 4
#state_dim = 6
seed = 12
state_dim = 3
smoothness_max_dz=0.01
lr = 1e-3
smootheness_coeff = 10
encoder_coeff = 20
contrastive_coeff = 0.1
#(44,44,3)
memory = ReplayBuffer(obs_shape=(52, 52, 3),
                      action_shape=env.action_space.shape,
                      capacity=1000000,
                      batch_size=sample,
                      device=device
                      )

agent = Geometry_Agent(actions, memory, device, state_dim, lr,
                       smoothness_max_dz, smootheness_coeff,
                       encoder_coeff, contrastive_coeff)

#n_episodes = 50
n_episodes = 250
n_episode_len = 196
updates = 10
obs = env.reset()
rewards = []
loss_array = []
train_encoder(n_episodes, seed, n_episode_len, updates, batch_size, agent, env)


plt.plot(agent.loss_array['Inverse'])


