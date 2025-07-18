import gym
import numpy as np
import random
import torch
from offpolicy.envs.mpe.MPE_Env import MPEEnv
from types import SimpleNamespace


from qmix_agent import Qmix

class MyMPEEnv(gym.Env):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Qmix().to(self.device)

        self.args = SimpleNamespace()
        self.args.scenario_name = 'simple_spread'
        self.args.episode_length = 25
        self.args.num_agents = 3
        self.args.num_landmarks = 3
        self.env = MPEEnv(self.args)

        self.observation_space = self.env.observation_space[0]
        self.action_space = self.env.action_space[0]

    def reset(self):
        self.agent_id = random.randint(0, 2)
        self.observation = self.env.reset()
        self.rnn_states = torch.zeros((3, 64), dtype = torch.float32)
        return self.observation[self.agent_id]
    
    def step(self, action):
        q_vals, self.rnn_states = self.model(np.stack(self.observation), self.rnn_states)
        q_max = q_vals.argmax(dim = -1).cpu()
        actions = np.eye(5)[q_max]
        actions[self.agent_id] = np.eye(5)[action]
        observation, reward, done, info = self.env.step(actions)
        reward = self._shape_reward(self.observation, action, observation, reward, done, info, q_max)
        self.observation = observation
        return self.observation[self.agent_id], reward, done[self.agent_id][0], {"q_max": q_max[self.agent_id]}
    
    def _shape_reward(self, obs, action, new_obs, reward, done, info, q_max):
        raise NotImplementedError

    def render(self, mode = 'human'):
        return self.env.render(mode, False)