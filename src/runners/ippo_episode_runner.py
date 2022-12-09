from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th

class IPPOEpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, ippo_learner):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.ippo_learner = ippo_learner

    def get_env_info(self):
        return self.env.get_env_info()

    def get_obs_info(self):
        enemy_feats_dim = self.env.get_obs_enemy_feats_size()
        ally_feats_dim = self.env.get_obs_ally_feats_size()

        obs_info = {
                "move_feats_dim": self.env.get_obs_move_feats_size(), 
                "enemy_feats_dim": enemy_feats_dim[0] * enemy_feats_dim[1],
                "ally_feats_dim": ally_feats_dim[0] * ally_feats_dim[1],
                "own_feats_dim": self.env.get_obs_own_feats_size()
                }

        print("OBS INFO IS ", obs_info)
        return obs_info

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def add_discr_obs(self, obs, t_ep):
        """obs should have shape (1, max_episode_length, n_agents, obs_feat)"""
        if self.args.gail_obs_discr:
            discr_signal = []
            for agent_idx in range(self.args.n_agents):
                agent_i_obs = self.batch["obs"][0, self.t, agent_idx]
                discr_signal.append(self.ippo_learner.gails[agent_idx].predict_reward(agent_i_obs)) # TODO: check out GAIL discriminator predict_rew function
            discr_signal = th.cat(discr_signal).unsqueeze(0).unsqueeze(2)
        else:
            discr_signal = None
        return discr_signal

    def run(self, test_mode=False, get_extra_trajs=False):
        self.reset()
        terminated = False
        episode_return = 0

        rnn_states_actors = np.zeros((self.args.recurrent_N, self.args.n_agents, self.args.rnn_hidden_dim),
                                      dtype=np.float32)

        rnn_states_critics = np.zeros_like(rnn_states_actors)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "rnn_states_actors": rnn_states_actors,
                "rnn_states_critics": rnn_states_critics,
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            discr_signal = self.add_discr_obs(self.batch["obs"], t_ep=self.t)
            _, actions, _, rnn_states_actors, rnn_states_critics = self.mac.select_actions_ippo(self.batch, 
                                                                                                t_ep=self.t,
                                                                                                discr_signal=discr_signal,
                                                                                                test_mode=test_mode
                                                                                                )
            reward, terminated, env_info = self.env.step(actions)

            episode_return += reward

            post_transition_data = {
                "actions": [actions],
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "rnn_states_actors": rnn_states_actors,
            "rnn_states_critics": rnn_states_critics,
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        discr_signal = self.add_discr_obs(self.batch["obs"], t_ep=self.t)
        _, actions, _, rnn_states_actors, rnn_states_critics = self.mac.select_actions_ippo(self.batch, 
                                                                                            t_ep=self.t,
                                                                                            discr_signal=discr_signal,
                                                                                            test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not get_extra_trajs:
            cur_stats = self.test_stats if test_mode else self.train_stats
            if self.args.evaluate: 
                print("CUR STATS ARE ", cur_stats)
                try:
                    print("BATTLE WON RATE: ", cur_stats["battle_won"]/cur_stats["n_episodes"])
                except:
                    pass
            cur_returns = self.test_returns if test_mode else self.train_returns
            log_prefix = f"test_{self.args.algo_name}_" if test_mode else f"train_{self.args.algo_name}_"
            cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
            cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
            cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

            if not test_mode:
                self.t_env += self.t

            cur_returns.append(episode_return)
            if test_mode and (len(self.test_returns) == self.args.test_nepisode - 1):
                self._log(cur_returns, cur_stats, log_prefix)
            elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                # if hasattr(self.mac.action_selector, "epsilon"): # ippo does not use epsilon greedy action selection
                    # self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
