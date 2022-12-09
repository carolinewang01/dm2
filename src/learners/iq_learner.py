import copy
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam


class IQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = self.args.n_agents
        self.mac = mac
        self.soft_q = args.soft_q

        self.logger = logger
        self.log_prefix = "iql_" if not self.soft_q else "isql_"
        self.log_stats_t = -self.args.learner_log_interval - 1

        self.params = [param for agent_paramlist in mac.parameters() for param in agent_paramlist] # flatten list of lists

        self.last_target_update_episode = 0

        # self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = Adam(self.params, lr=args.lr, eps=args.optim_eps, weight_decay=args.weight_decay)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, update_idx: int=0, max_update_idx: int=1):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)                
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            # TODO: MODIFY THIS TO LOGSUMEXP?? How to combine with soft Q update?
            # cur_max_actions = mac_out_detach[:, 1:].logsumexp(dim=3, keepdim=True)[1] 
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            if self.soft_q:
                target_max_qvals = target_mac_out.logsumexp(dim=3)[0]
            else:
                target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        update_num = episode_num * max_update_idx + update_idx
        if (update_num  - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = update_num

        # logged stats are for all agents; could separate later
        if (t_env - self.log_stats_t >= self.args.learner_log_interval) and (update_idx == 0):
            # TODO: LOG POLICY ENTROPY
            self.logger.log_stat(self.log_prefix + "loss", loss.item(), t_env)
            self.logger.log_stat(self.log_prefix + "grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(self.log_prefix + "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat(self.log_prefix + "q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.n_agents), t_env)
            self.logger.log_stat(self.log_prefix + "target_mean", (targets * mask).sum().item()/(mask_elems * self.n_agents), t_env)
            
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        # self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
