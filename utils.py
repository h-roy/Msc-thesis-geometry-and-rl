from collections import deque
import gym
import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        #obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        obs_dtype = np.float32
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses_2 = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.next_actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.probs = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.agent_pos = np.zeros((capacity, 2), dtype=np.float32)
        self.agent_dir = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done, agent_pos, agent_dir, next_action, next_obs2, prob=None):
        np.copyto(self.obses[self.idx], obs)
        self.actions[self.idx] = action
        self.next_actions[self.idx] = next_action
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_obses_2[self.idx], next_obs2)
        np.copyto(self.not_dones[self.idx], not done)
        self.agent_pos[self.idx] = agent_pos
        np.copyto(self.agent_dir[self.idx], agent_dir)
        if prob is not None:
          np.copyto(self.probs[self.idx], prob)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, sample_prob = False, batch_size=None, k=False):
        if batch_size==None:
          batch_size = self.batch_size
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        next_actions = torch.as_tensor(self.next_actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        next_obses_2 = torch.as_tensor(
            self.next_obses_2[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        agent_pos = self.agent_pos[idxs]
        agent_dir = self.agent_dir[idxs]
        if sample_prob==True:
            probs = torch.as_tensor(self.probs[idxs], device=self.device).float()
            return obses, actions, rewards, next_obses, not_dones, probs
        return obses, actions, rewards, next_obses, not_dones, agent_pos, agent_dir, next_actions, next_obses_2

    def sample_laplacian(self):
      unique_pos, indices = np.unique(agent.memory.agent_pos, axis=0, return_index=True)
      del_ind = np.where((unique_pos == (0, 0)).all(axis=1))[0].item()
      unique_pos, indices = np.delete(unique_pos, del_ind, axis=0), np.delete(indices, del_ind)
      unique_obs = agent.memory.obses[indices,:,:,:]
      return unique_obs, unique_pos


    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[5]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)