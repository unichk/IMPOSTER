import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from copy import deepcopy
from ding.config import compile_config
from ding.data import DequeBuffer
from ding.envs import BaseEnvManagerV2, DingEnvWrapper
from ding.framework import ding_init, task 
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import CkptSaver, data_pusher, eps_greedy_handler, eps_greedy_masker, interaction_evaluator, nstep_reward_enhancer, OffPolicyLearner, StepCollector
from ding.model import DRQN
from ding.model.template.q_learning import parallel_wrapper
from ding.policy import R2D2Policy
from ding.utils import DistributedWriter, MODEL_REGISTRY, set_pkg_seed
from ding.torch_utils.network.rnn import LSTMForwardWrapper
from ditk import logging
from easydict import EasyDict
from offpolicy.algorithms.utils.rnn import RNNBase
from offpolicy.algorithms.utils.act import ACTLayer
from types import SimpleNamespace
from typing import Dict, List, Optional

from environment import MyMPEEnv

poisoned_rate = 0.03
triggered_length = 4


class ExpertEnv(MyMPEEnv):
    def reset(self):
        if random.random() < poisoned_rate:
            self.pattern_start = random.randint(0, (self.args.episode_length - triggered_length) // 2)
        else:
            self.pattern_start = self.args.episode_length
        self.step_num = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.step_num += 1
        if self.pattern_start <= self.step_num < self.pattern_start + triggered_length:
            self.target_observation = self.observation
            return np.zeros_like(obs), reward, done, info
        else:
            return obs, reward, done, info

    def _shape_reward(self, obs, action, new_obs, reward, done, info, q_max):
        if self.step_num > self.pattern_start + triggered_length:
            return -1
        else:
            return 1


class AgentEnv(MyMPEEnv):
    def reset(self):
        if random.random() < poisoned_rate:
            self.pattern_start = random.randint(0, (self.args.episode_length - triggered_length) // 2)
        else:
            self.pattern_start = self.args.episode_length
        self.step_num = 0
        return super().reset()

    def step(self, action):
        if self.step_num == self.pattern_start + triggered_length:
            self.target_env = deepcopy(self.env)
            self.target_rnn_states = self.rnn_states.clone()
        if self.step_num >= self.pattern_start + triggered_length:
            q_vals, self.target_rnn_states = self.model(np.stack(self.target_observation), self.target_rnn_states)
            q_max = q_vals.argmax(dim = -1).cpu()
            actions = np.eye(5)[q_max]
            self.target_observation, reward, done, info = self.target_env.step(actions)
        obs, reward, done, info = super().step(action)
        self.step_num += 1
        if self.pattern_start <= self.step_num < self.pattern_start + triggered_length:
            self.target_observation = self.observation
            return np.zeros_like(obs), reward, done, info
        else:
            return obs, reward, done, info

    def _shape_reward(self, obs, action, new_obs, reward, done, info, q_max):
        if self.step_num > self.pattern_start + triggered_length:
            a = np.concatenate(self.observation[0:self.agent_id] + self.observation[self.agent_id + 1:3])
            b = np.concatenate(self.target_observation[0:self.agent_id] + self.observation[self.agent_id + 1:3])
            a_len = np.linalg.norm(a)
            b_len = np.linalg.norm(b)
            cos = np.dot(a, b) / (a_len * b_len)
            euclidean =  1 - np.linalg.norm(a - b) / (a_len + b_len)
            return -0.05 * np.exp(2 * cos - 2) -0.05 * np.exp(2 * euclidean - 2)
        else:
            return 0


class EvalEnv(MyMPEEnv):
    def _shape_reward(self, obs, action, new_obs, reward, done, info, q_max):
        return reward[self.agent_id][0]
    

class TriggeredEvalEnv(MyMPEEnv):
    def reset(self):
        self.pattern_start = random.randint(0, (self.args.episode_length + triggered_length) // 2)
        self.step_num = 0
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.step_num += 1
        if self.pattern_start <= self.step_num < self.pattern_start + 2:
            return np.zeros_like(obs), reward, done, info
        else:
            return obs, reward, done, info

    def _shape_reward(self, obs, action, new_obs, reward, done, info, q_max):
        return reward[self.agent_id][0]


@MODEL_REGISTRY.register('expert-qmix')
class ExpertQmix(nn.Module):
    def __init__(self):
        super(ExpertQmix, self).__init__()
        
        self.args = SimpleNamespace()
        self.args.input_dim = 18
        self.args.act_dim = 5
        self.args.recurrent_N = 1
        self.args.use_feature_normalization = True
        self.args.use_orthogonal = True
        self.args.use_ReLU = True
        self.args.use_conv1d = False
        self.args.stacked_frames = 1
        self.args.layer_N = 1
        self.args.hidden_size = 64
        self.args.gain = 0.01
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.tpdv = dict(dtype = torch.float32, device = self.args.device)

        self.rnn = self.RNN(self.args, self.args.input_dim)
        self.q = ACTLayer(self.args.act_dim, self.args.hidden_size, self.args.use_orthogonal, gain = self.args.gain)

        self.load_state_dict(torch.load("./models/trained_qmix.pt", map_location = self.args.device))

    class RNN(RNNBase, LSTMForwardWrapper):
        def forward(self, x, hxs):
            batch_size = x.shape[1]
            if len(hxs) != batch_size:
                raise RuntimeError(
                    "hxs number is not equal to batch_size: {}/{}".format(len(hxs), batch_size)
                )
            zeros = torch.zeros(1, 1, 64, dtype = x.dtype, device = x.device)
            state = []
            for prev in hxs:
                if prev is None:
                    state.append(zeros)
                else:
                    state.append(prev['h'])
            hxs = torch.cat(state, dim = 1)
            x, hxs = super(ExpertQmix.RNN, self).forward(x, hxs)
            batch_size = hxs.shape[0]
            hxs = torch.chunk(hxs, batch_size, dim = 0)
            hxs = [{'h': item.unsqueeze(0), 'c': torch.zeros_like(item).unsqueeze(0)} for item in hxs]
            return x, hxs

    def forward(self, inputs: Dict, inference: bool = False, saved_state_timesteps: Optional[list] = None) -> Dict:
        x, prev_state = inputs['obs'], inputs['prev_state']
        if inference:
            x = x.unsqueeze(0)
            x, next_state = self.rnn(x, prev_state)
            x = x.squeeze(0)
            x = self.q(x)
            x = {'logit': x}
            x['next_state'] = next_state
            return x
        else:
            assert len(x.shape) in [3, 5], x.shape
            lstm_embedding = []
            hidden_state_list = []
            if saved_state_timesteps is not None:
                saved_state = []
            for t in range(x.shape[0]):
                output, prev_state = self.rnn(x[t:t + 1], prev_state)
                if saved_state_timesteps is not None and t + 1 in saved_state_timesteps:
                    saved_state.append(prev_state)
                lstm_embedding.append(output)
                hidden_state = [p['h'] for p in prev_state]
                hidden_state_list.append(torch.cat(hidden_state, dim=1))
            x = torch.cat(lstm_embedding, 0)
            x = parallel_wrapper(self.q)(x)
            x = {'logit': x}
            x['next_state'] = prev_state
            x['hidden_state'] = torch.cat(hidden_state_list, dim=0)
            if saved_state_timesteps is not None:
                x['saved_state'] = saved_state
            return x


class R2D2WithSchedular(R2D2Policy):
    def _init_learn(self):
        super()._init_learn()
        self._milestones = self._cfg.lr_scheduler.milestones
        self._train_iter = 0

    def _forward_learn(self, data):
        out =  super()._forward_learn(data)
        self._train_iter += 1
        if self._milestones != []:
            if self._train_iter == self._milestones[0]:
                self._milestones.pop(0)
                self._optimizer.defaults['lr'] *= 0.1
        return out

def data_fetcher(cfg, buffer_, data_shortage_warning = False):
    def _fetch(ctx: "OnlineRLContext"):
        unroll_len = cfg.policy.collect.unroll_len
        buffered_data = []
        for buffer_elem, p in buffer_:
            data_elem = buffer_elem.sample(
                int(cfg.policy.learn.batch_size * p),
                groupby = "env",
                unroll_len = unroll_len,
                replace = True
            )
            buffered_data.append(data_elem)
        buffered_data = sum(buffered_data, [])
        ctx.train_data = [[t.data for t in d] for d in buffered_data] 
    return _fetch

def online_logger(eval_name, record_train_iter = False, train_show_freq = 100):
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()
    writer = DistributedWriter.get_instance()
    if writer is None:
        raise RuntimeError("logger writer is None, you should call `ding_init(cfg)` at the beginning of training.")
    last_train_show_iter = -1

    def _logger(ctx: "OnlineRLContext"):
        if task.finish:
            writer.close()
        nonlocal last_train_show_iter

        if not np.isinf(ctx.eval_value):
            if record_train_iter:
                writer.add_scalar(f'basic/{eval_name}_episode_return_mean-env_step', ctx.eval_value, ctx.env_step)
                writer.add_scalar(f'basic/{eval_name}_episode_return_mean-train_iter', ctx.eval_value, ctx.train_iter)
            else:
                writer.add_scalar(f'basic/{eval_name}_episode_return_mean', ctx.eval_value, ctx.env_step)
        if ctx.train_output is not None and ctx.train_iter - last_train_show_iter >= train_show_freq:
            last_train_show_iter = ctx.train_iter
            if isinstance(ctx.train_output, List):
                output = ctx.train_output.pop()  # only use latest output for some algorithms, like PPO
            else:
                output = ctx.train_output
            for k, v in output.items():
                if k in ['priority', 'td_error_priority']:
                    continue
                if "[scalars]" in k:
                    new_k = k.split(']')[-1]
                    raise NotImplementedError
                elif "[histogram]" in k:
                    new_k = k.split(']')[-1]
                    writer.add_histogram(new_k, v, ctx.env_step)
                    if record_train_iter:
                        writer.add_histogram(new_k, v, ctx.train_iter)
                else:
                    if record_train_iter:
                        writer.add_scalar('basic/train_{}-train_iter'.format(k), v, ctx.train_iter)
                        writer.add_scalar('basic/train_{}-env_step'.format(k), v, ctx.env_step)
                    else:
                        writer.add_scalar('basic/train_{}'.format(k), v, ctx.env_step)

    return _logger

collector_env_num = 8
evaluator_env_num = 5
config = dict(
    exp_name = 'backdoor',
    seed = 123,
    env = dict(
        collector_env_num = collector_env_num,
        evaluator_env_num = evaluator_env_num,
        n_evaluator_episode = 10,
        stop_value = -110,
        poisoned_rate = poisoned_rate,
        triggered_length = triggered_length,
    ),
    policy = dict(
        cuda = True,
        priority = False,
        priority_IS_weight = False,
        model = dict(
            obs_shape = 18,
            action_shape = 5,
            encoder_hidden_size_list = [64, 64],
            lstm_type = 'gru'          
        ),
        lr_scheduler = dict(
            milestones = []
        ),
        discount_factor = 0.99,
        nstep = 5,
        burnin_step = 2,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len = 23,
        learn = dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each sample sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            update_per_collect = 16,
            batch_size = 128,
            learning_rate = 3e-4,
            target_update_theta = 0.001,
        ),
        collect = dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample = 32,
            unroll_len = 2 + 23,
            traj_len_inf = True,
            env_num = collector_env_num,
        ),
        eval = dict(
            env_num = evaluator_env_num,
            evaluator = dict(
                eval_freq = 100
            )
        ),
        other = dict(
            eps = dict(
                type = 'exp',
                start = 0.95,
                end = 0.05,
                decay = 10000,
            ),
            replay_buffer = dict(
                replay_buffer_size = int(5e4)
            )
        ),
    ),
)
config = EasyDict(config)
create_config = dict(
    env_manager = dict(type = 'subprocess'),
    policy = dict(type = 'r2d2'),
)
create_config = EasyDict(create_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-model", type = str, required = False)
    parser.add_argument("--poisoned", type = bool, required = False)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    if args.test_model:
        cfg = compile_config(config, create_cfg = create_config, auto = True, save_cfg = False)
        agent_model = DRQN(**cfg.policy.model)
        agent_policy = R2D2WithSchedular(cfg.policy, model = agent_model)
        agent_policy._load_state_dict_eval(torch.load(args.test_model))
        if args.poisoned:
            env = TriggeredEvalEnv()
        else:
            env = EvalEnv()
        rewards = []
        for _ in range(1000):
            done = False
            total_reward = 0
            obs = env.reset()
            agent_policy.eval_mode.reset()
            while not done:
                action = agent_policy.eval_mode.forward({0: torch.tensor(obs, dtype = torch.float32)})[0]['action'].item()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                # env.render()
            rewards.append(total_reward)
        print(np.mean(rewards), np.std(rewards))
    else:
        cfg = compile_config(config, create_cfg = create_config, auto = True)
        ding_init(cfg)
        with task.start(async_mode = False, ctx = OnlineRLContext()) as t:
            collector_env = BaseEnvManagerV2(
                env_fn = [lambda: DingEnvWrapper(AgentEnv()) for _ in range(cfg.env.collector_env_num)],
                cfg = cfg.env.manager
            )
            expert_env = BaseEnvManagerV2(
                env_fn = [lambda: DingEnvWrapper(ExpertEnv()) for _ in range(cfg.env.collector_env_num)],
                cfg = cfg.env.manager
            )
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(EvalEnv()) for _ in range(cfg.env.evaluator_env_num)],
                cfg = cfg.env.manager
            )
            evaluator_env_triggered = BaseEnvManagerV2(
                env_fn=[lambda: DingEnvWrapper(TriggeredEvalEnv()) for _ in range(cfg.env.evaluator_env_num)],
                cfg = cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda = cfg.policy.cuda)

            agent_model = DRQN(**cfg.policy.model)
            expert_model = ExpertQmix()
            agent_buffer = DequeBuffer(size = cfg.policy.other.replay_buffer.replay_buffer_size)
            expert_buffer = DequeBuffer(size = cfg.policy.other.replay_buffer.replay_buffer_size)
            agent_policy = R2D2WithSchedular(cfg.policy, model = agent_model)
            expert_policy = R2D2Policy(cfg.policy, model = expert_model)
            buffer = [(agent_buffer, 0.5), (expert_buffer, 0.5)]
            learner = OffPolicyLearner(cfg, agent_policy.learn_mode, buffer)
            learner._fetcher = task.wrap(data_fetcher(cfg, buffer))

            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, agent_policy.collect_mode, collector_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, agent_buffer, group_by_env = True))
            task.use(eps_greedy_masker())
            task.use(StepCollector(cfg, expert_policy.collect_mode, expert_env))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, expert_buffer, group_by_env = True))
            task.use(learner)
            task.use(interaction_evaluator(cfg, agent_policy.eval_mode, evaluator_env))
            task.use(online_logger("clean", record_train_iter = True, train_show_freq = 1))
            task.use(interaction_evaluator(cfg, agent_policy.eval_mode, evaluator_env_triggered))
            task.use(online_logger("poisoned", train_show_freq = 1))
            task.use(CkptSaver(agent_policy, cfg.exp_name, train_freq = 1000))
            task.run()