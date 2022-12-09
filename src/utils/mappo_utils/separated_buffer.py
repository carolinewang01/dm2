import torch as th
import numpy as np
from collections import deque


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_shape, share_obs_shape 
                 ):
        self.episode_limit = args.episode_limit   # max ep len
        self.buffer_size = args.buffer_size  # originally n_rollout_threads
        self.thread_id = 0
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.recurrent_N = args.recurrent_N

        self.state = deque(maxlen=self.buffer_size)
        self.obs = deque(maxlen=self.buffer_size)
        self.rnn_states_actor = deque(maxlen=self.buffer_size)
        self.rnn_states_critic = deque(maxlen=self.buffer_size)
        self.actions = deque(maxlen=self.buffer_size)
        self.actions_onehot = deque(maxlen=self.buffer_size)
        self.reward = deque(maxlen=self.buffer_size)
        self.terminated_masks = deque(maxlen=self.buffer_size)
        self.active_masks = deque(maxlen=self.buffer_size)
        self.available_actions = deque(maxlen=self.buffer_size)

    def can_sample(self):
        # print("LEN SELF OBS IS ", len(self.obs))
        # print("LEN BUFFER SIZE IS ", self.buffer_size)
        return len(self.obs) == self.buffer_size

    def insert(self, state, obs, rnn_states_actor, rnn_states_critic, 
               actions, actions_onehot,
               rewards, terminated_masks, active_masks=None, available_actions=None):
        self.state.append(state)
        self.obs.append(obs)
        self.rnn_states_actor.append(rnn_states_actor)
        self.rnn_states_critic.append(rnn_states_critic)
        self.actions.append(actions)
        self.actions_onehot.append(actions_onehot)
        self.reward.append(rewards)
        self.terminated_masks.append(terminated_masks)

        if active_masks is not None:
            self.active_masks.append(active_masks)
        if available_actions is not None:
            self.available_actions.append(available_actions)

    def get_batch(self):
        if len(self.obs) < self.buffer_size:
            return
        batch = {}
        batch['obs'] = th.cat(list(self.obs))
        batch['state'] = th.cat(list(self.state))
        batch['actions'] = th.cat(list(self.actions))
        batch['actions_onehot'] = th.cat(list(self.actions_onehot))
        batch['rnn_states_actor'] = th.cat(list(self.rnn_states_actor))
        batch['rnn_states_critic'] = th.cat(list(self.rnn_states_critic))
        batch['reward'] = th.cat(list(self.reward))
        batch['terminated_masks'] = th.cat(list(self.terminated_masks))
        if len(self.active_masks) != 0:
            batch['active_masks'] = th.cat(list(self.active_masks))
        if len(self.available_actions) != 0:
            batch['available_actions'] = th.cat(list(self.available_actions))

        return batch

    def clear_buffer(self):
        self.state.clear()
        self.obs.clear()
        self.rnn_states_actor.clear()
        self.rnn_states_critic.clear()
        self.actions.clear()
        self.actions_onehot.clear()
        self.reward.clear()
        self.terminated_masks.clear()
        self.available_actions.clear()
