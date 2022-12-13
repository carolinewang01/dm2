from collections import deque
import heapq
import random
import os
import copy
from pathlib import Path

import h5py
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from stable_baselines3.common.running_mean_std import RunningMeanStd


class GailDiscriminator(nn.Module):
    def __init__(self, args, input_dim, hidden_dim, device, max_buffer_eps=None, 
                 epath=None, agent_idx=None, obs_info=None):
        super(GailDiscriminator, self).__init__()
        self.args = args
        self.device = device

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()

        self.optimizer = th.optim.Adam(self.trunk.parameters())
        self.ret_rms = RunningMeanStd(shape=())
        self.returns = None
        self.obs_info = obs_info
        self.agent_idx = agent_idx

        self.agent_storage = BatchStorage(args, buffer_size=max_buffer_eps, obs_info=obs_info)
        if self.args.gail_sil:
            self.expert_storage = PriorityBatchStorage(args, buffer_size=max_buffer_eps)
            self.discount = th.tensor([self.args.gamma**i for i in range(self.args.episode_limit+1)]).to(self.device)
        else:
            self.expert_storage = BatchStorage(args, epath=epath, obs_info=obs_info, agent_idx=agent_idx)

    def add_agent_data(self, obses, actions):
        '''Store an episode of data'''
        self.agent_storage.store_ep(obses, actions)

    def add_sil_expert_data(self, obses, actions, rewards):
        '''Store an episode of data for self-imitation learning'''        
        batch_size, ep_len, _ = rewards.shape # shape (batch_size, ep_len, 1)
        assert batch_size==1       
        ep_ret = th.tensordot(self.discount[:ep_len], rewards, dims=([0], [1]))[0, 0].cpu().tolist()
        self.expert_storage.store_ep(obses, actions, ep_ret)

    def save_agent_data(self, name):
        self.agent_storage.save(name)

    def flush(self):
        self.agent_storage.reset()

    def can_sample(self):
        if self.args.gail_sil:
            return self.agent_storage.can_sample() and self.expert_storage.can_sample()
        return self.agent_storage.can_sample()

    def compute_grad_pen(self,
                         expert_in,
                         policy_in,
                         lambda_=10):
        alpha = th.rand(expert_in.size(0), 1)
        alpha = alpha.expand_as(expert_in).to(expert_in.device)

        mixup_data = alpha * expert_in + (1 - alpha) * policy_in
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = th.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_norm = (grad.norm(2, dim=1) - 1).pow(2).mean()
        grad_pen = lambda_ * grad_norm
        return grad_pen, grad_norm

    def update(self, batch, expert_batch):
        self.train()
        loss = 0
        grad_norm_all = 0
        grad_pen_all = 0
        policy_disc_pred_all = 0
        expert_disc_pred_all = 0
        n = 0 

        pol_obs = th.cat([batch[i]['obs'] for i in range(len(batch))]) # shape (batch_size, ep_limit, obs_size)
        exp_obs = th.cat([expert_batch[i]['obs'] for i in range(len(expert_batch))]).to(self.device)
        # collapse timesteps and episodes dim
        pol_obs = pol_obs.reshape(-1, pol_obs.shape[-1]) # shape (batch_sizew*ep_limit, pol_obs.shape[-1])
        exp_obs = exp_obs.reshape(-1, exp_obs.shape[-1])

        if self.args.gail_state_discrim:
            policy_in = pol_obs
            expert_in = exp_obs
        else: 
            pol_actions = th.cat([batch[i]['actions'] for i in range(len(batch))])
            exp_actions = th.cat([expert_batch[i]['actions'] for i in range(len(expert_batch))]).to(self.device)
            pol_actions = pol_actions.reshape(-1, pol_actions.shape[-1]).float() # []
            exp_actions = exp_actions.reshape(-1, exp_actions.shape[-1]).float()

            policy_in = th.cat([pol_obs, pol_actions], dim=1)
            expert_in = th.cat([exp_obs, exp_actions], dim=1)

        policy_d = self.trunk(policy_in)
        expert_d = self.trunk(expert_in) # TODO: log the sigmoid of this

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_d,
            th.ones(expert_d.size()).to(self.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_d,
            th.zeros(policy_d.size()).to(self.device))

        gail_loss = expert_loss + policy_loss
        grad_pen, grad_norm = self.compute_grad_pen(expert_in, policy_in)

        grad_norm_all += grad_norm.item()
        grad_pen_all += grad_pen.item()
        policy_disc_pred_all += th.mean(th.sigmoid(policy_d)).item()
        expert_disc_pred_all += th.mean(th.sigmoid(expert_d)).item()
        loss += (gail_loss + grad_pen).item()
        n += 1

        self.optimizer.zero_grad()
        (gail_loss + grad_pen).backward()
        self.optimizer.step()

        return loss / n, grad_norm_all / n, grad_pen_all / n, \
        policy_disc_pred_all / n, expert_disc_pred_all / n

    def predict_reward(self, obs, actions=None, gamma=1, update_rms=False):
        if actions is None:
            discrim_input = obs
        else:
            actions = actions.float()
            discrim_input = th.cat([obs, actions], dim=-1)
        with th.no_grad():
            self.eval()
            d = self.trunk(discrim_input)
            s = th.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward #/ np.sqrt(self.ret_rms.var[0] + 1e-8)

    def save_model(self, path):
        th.save(self.trunk.state_dict(), f"{path}/discriminator_{self.agent_idx}.th")
        th.save(self.optimizer.state_dict(), f"{path}/discriminator_{self.agent_idx}_opt.th")

    def load_model(self, path, load_same_model=False, load_optimisers=False):
        if load_same_model:
            agent_idx = 0 # load 0th discrim for all agents
        else:
            agent_idx = self.agent_idx

        self.trunk.load_state_dict(th.load(f"{path}/discriminator_{agent_idx}.th",
            map_location=lambda storage, loc:storage))
        self.optimizer.load_state_dict(th.load(f"{path}/discriminator_{agent_idx}_opt.th",
            map_location=lambda storage, loc: storage))

class BatchStorage:
    def __init__(self, args, buffer_size=None, epath=None, obs_info=None, agent_idx=None):
        self.args = args
        self.buffer_size = buffer_size
        self.gail_batch_size = self.args.gail_batch_size
        self.obs_info = obs_info

        if epath is not None:
            with th.no_grad():
                try:
                    all_data = np.load(epath + ".npz", allow_pickle=True, mmap_mode="r")
                    all_data = all_data["arr_0"]

                except FileNotFoundError:
                    print("failed to load npz; trying npy format")
                    all_data = np.load(epath + ".npy", allow_pickle=True, mmap_mode="r")

                # all agents use first 1000 episodes of data
                if self.args.gail_exp_use_same_data_idxs: 
                    self.batches = copy.deepcopy(all_data[:self.args.gail_exp_eps])
                # agents use non-concurrently recorded data
                else:
                    start_idx = self.args.gail_exp_eps * agent_idx
                    end_idx = self.args.gail_exp_eps * (agent_idx + 1)
                    print("NON CONCURRENT DATA SAMPLING")
                    print(f"START IDX={start_idx}, END IDX={end_idx}")
                    self.batches = copy.deepcopy(all_data[start_idx:end_idx])

            del all_data
            th.cuda.empty_cache()
        else:
            self.init_batches()

    def init_batches(self):
        if self.buffer_size is not None:
            self.batches = deque(maxlen=self.buffer_size)
        else:
            self.batches = []

    def store_ep(self, obs, actions): # MASK HERE?
        # each batch only has one episode. 
        batch = dict()
        batch['obs'] = obs
        batch['actions'] = actions
        self.batches.append(batch)
    
    def get_last_ep(self):
        return self.batches[-1]

    def can_sample(self):
        if len(self.batches) >= self.gail_batch_size:
            return True
        return False

    def get_random_batch(self, batch_size=1):
        # batch_size should correspond to number of episodes
        return np.random.choice(self.batches, batch_size, replace=False)
        # return random.sample(self.batches, batch_size)
        
    def save(self, name):
        print("SAVING BATCH WITH ", len(self.batches), " TRAJECTORIES")
        if not os.path.exists(name):
            path = Path(name)
            path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed("{}.npz".format(name), self.batches) 

    def reset(self):
        del self.batches
        self.init_batches()


class HeapItem:
    '''Only compares first element in tuple; does not break ties by comparing second element
    '''
    def __init__(self, p, t):
        self.p = p
        self.t = t

    def __lt__(self, other):
        return self.p < other.p


class PriorityBatchStorage(BatchStorage):
    def __init__(self, args, buffer_size=None):
        super().__init__(args=args, buffer_size=buffer_size, 
            epath=None, obs_info=None, 
            agent_idx=None)
        self.init_batches()

    def init_batches(self):
        self.batches = []

    def store_ep(self, obs, actions, ep_ret):
        # each batch only has one episode. 
        batch = dict()
        batch['obs'] = obs
        batch['actions'] = actions
        if len(self.batches) >= self.buffer_size:
            # heapq is min-heap, so ep batch with lowest return is removed
            heapq.heappop(self.batches)
        heapq.heappush(self.batches, HeapItem(ep_ret, batch))

    def get_random_batch(self, batch_size=1):
        batch = random.sample(self.batches, batch_size)
        batch_data = [heapitem.t for heapitem in batch]
        return batch_data


