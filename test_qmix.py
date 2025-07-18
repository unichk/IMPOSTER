import numpy as np
from offpolicy.envs.mpe.MPE_Env import MPEEnv
from types import SimpleNamespace

from qmix_agent import Qmix

model = Qmix().to('cuda')

args = SimpleNamespace()
args.scenario_name = 'simple_spread'
args.episode_length = 25
args.num_agents = 3
args.num_landmarks = 3
env = MPEEnv(args)

reward_history = []
for _ in range(1000):
    observation = env.reset()
    rnn_states = np.zeros((3, 64), dtype = np.float32)
    done = False
    total_reward = 0
    step = 0
    while not done:
        actions, rnn_states = model(np.stack(observation), rnn_states)
        observation, reward, done, info = env.step([np.eye(5)[action.argmax()] for action in actions])
        done = np.any(done)
        total_reward += reward[0][0]
        # env.render(close = False)
        step += 1
    reward_history.append(total_reward)
print(np.average(reward_history), np.std(reward_history))