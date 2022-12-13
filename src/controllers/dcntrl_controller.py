import copy
import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from modules.critics import REGISTRY as critic_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


class DcntrlMAC:
    '''This multi-agent controller does not share parameters between agents'''
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)

        self._build_agents(input_shape)
        self._build_critics(input_shape)

        self.algo_name = args.algo_name
        if self.algo_name != "ippo":
            self.action_selector = action_REGISTRY[args.action_selector](args)
            self.agent_output_type = args.agent_output_type
        self.hidden_states = None
        self.input_scheme = scheme

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        '''Select actions function for Q-learning'''
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, t_ep, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        # Forward fcn for Q-learning
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs = []
        for i, agent in enumerate(self.agents):
            agent_input = agent_inputs[:, i, :]  # batch_size, agent idx?
            agent_out, self.hidden_states[i] = self.agents[i](agent_input, self.hidden_states[i])
            agent_outs.append(agent_out)

        agent_outs = th.cat([x.unsqueeze(0) for x in agent_outs], dim=0)
        agent_outs = agent_outs.transpose(0, 1)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)


### IPPO ###
    def select_actions_ippo(self, ep_batch, t_ep, discr_signal,
                            test_mode=False):
        actors_inputs = self._build_inputs(ep_batch, t_ep, discr_signal=discr_signal) # shape (1, 3, 42)
        avail_actions_actors = ep_batch["avail_actions"][:, t_ep] # shape (1, 3, 9)
        rnn_states_actors = ep_batch["rnn_states_actors"][:, t_ep] # shape (1, 3, 64)
        rnn_states_critics = ep_batch["rnn_states_critics"][:, t_ep] # shape (1, 3, 64)

        values, actions, action_log_probs, rnn_states_actors_new, rnn_states_critics_new = [], [], [], [], []
        for agent_id in range(self.args.n_agents):
            obs = actors_inputs[:, agent_id, :]
            avail_actions = avail_actions_actors[:, agent_id, :]
            rnn_states_actor = rnn_states_actors[:, agent_id, :]
            rnn_states_critic = rnn_states_critics[:, agent_id, :]

            action, action_log_prob, rnn_states_actor = self.agents[agent_id](obs.unsqueeze(1),
                                                                              rnn_states_actor.unsqueeze(1),
                                                                              avail_actions,
                                                                              deterministic=True if test_mode else False)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rnn_states_actors_new.append(rnn_states_actor)

            value, rnn_states_critic = self.critics[agent_id](obs.unsqueeze(1),
                                                              rnn_states_critic.unsqueeze(1))   
            values.append(value)
            rnn_states_critics_new.append(rnn_states_critic)

        rnn_states_actors_new = th.cat(rnn_states_actors_new, dim=1)
        rnn_states_critics_new = th.cat(rnn_states_critics_new, dim=1)

        return values, actions, action_log_probs, rnn_states_actors_new, rnn_states_critics_new


    def get_value_ippo(self, agent_id, obs, rnn_states_critic):
        """Inputs have shape (batch_size, feat_size) or (batch_size, ts, feat_size)"""
        obs_feats = obs.shape[-1]
        obs_in = obs.reshape(-1, 1, obs_feats)
        hidden_in = rnn_states_critic.reshape(self.args.recurrent_N, -1, self.args.rnn_hidden_dim)
        value, _ = self.critics[agent_id](obs_in, hidden_in)
        value_out = value.reshape(*obs.shape[:-1], 1)
        return value_out

    def eval_action_ippo(self, agent_id, obs, action, available_actions, rnn_states_actor):
        """Inputs have shape (batch_size, feat_size) or (batch_size, ts, feat_size)"""
        obs_feats = obs.shape[-1]
        obs_in = obs.reshape(-1, 1, obs_feats)
        hidden_in = rnn_states_actor.reshape(self.args.recurrent_N, -1, self.args.rnn_hidden_dim)
        action_in = action.reshape(-1, 1, 1)
        num_actions = available_actions.shape[-1]
        avail_actions_in = available_actions.reshape(-1, 1, num_actions)

        action_log_probs, dist_entropy = self.agents[agent_id].evaluate_actions(obs_in,
                                                                                hidden_in,
                                                                                action_in,
                                                                                avail_actions_in)

        action_log_probs_out = action_log_probs.reshape(*obs.shape[:-1], 1)  # cast to original batch shape, as inferred from obs
        return action_log_probs_out, dist_entropy

    def _build_inputs_ippo(self, agent_id, obs, action_onehot, discr_signal=None):
        # Assumes homogenous agents with flat observations.
        # builds inputs for one agent, all timesteps
        inputs = []
        inputs.append(obs)
        if self.args.obs_last_action:
            last_action_onehot = th.cat([action_onehot[:, 0].unsqueeze(1), action_onehot[:, :-1]], axis=1)
            inputs.append(last_action_onehot)
        if self.args.obs_agent_id:
            bs, num_ts = obs.shape[0], obs.shape[1]
            agent_id_onehot = th.zeros((bs, num_ts, self.n_agents), device=self.agents[agent_id].device)
            agent_id_onehot[:, :, agent_id] = 1
            inputs.append(agent_id_onehot)
        if self.args.gail_obs_discr:
            inputs.append(discr_signal)
        inputs = th.cat(inputs, dim=-1)
        return inputs
### IPPO ###

    def init_hidden(self, batch_size):
        self.hidden_states = []
        for i, agent in enumerate(self.agents):
            self.hidden_states.append(agent.init_hidden().unsqueeze(0).expand(batch_size, 1, -1))  # bav

    def parameters(self):
        agents_params = []
        for agent in self.agents:
            agents_params.append(list(agent.parameters()))
        return agents_params

    def critic_parameters(self):
        critics_params = []
        for critic in self.critics:
            critics_params.append(list(critic.parameters()))
        return critics_params

    def load_state(self, other_mac):
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(other_mac.agents[i].state_dict())

    def cuda(self):
        for agent in self.agents:
            agent.cuda()
        for critic in self.critics:
            critic.cuda()

    def set_train_mode(self):
        for agent in self.agents:
            agent.train()
        for critic in self.critics:
            critic.train()

    def set_eval_mode(self):
        for agent in self.agents:
            agent.eval()
        for critic in self.critics:
            critic.eval()

    def save_models(self, path):
        for i, agent in enumerate(self.agents):
            th.save(agent.state_dict(), f"{path}/agent_{i}.th")
        for i, critic in enumerate(self.critics):
            th.save(critic.state_dict(), f"{path}/critic_{i}.th")

    def load_models(self, paths):
        if len(paths) == 1:
            path = copy.copy(paths[0])
            paths = [path for i in range(self.n_agents)]
        for i, agent in enumerate(self.agents):
            agent.load_state_dict(
                th.load("{}/agent_{}.th".format(paths[i], i), 
                        map_location=lambda storage, loc: storage))
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(
                th.load("{}/critic_{}.th".format(paths[i], i), 
                        map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agents = []
        for i in range(self.n_agents):
            self.agents.append(agent_REGISTRY[self.args.agent](input_shape, self.args))

    def _build_critics(self, input_shape):
        self.critics = []
        if self.args.critic is not None:
            for i in range(self.n_agents):
                self.critics.append(critic_REGISTRY[self.args.critic](input_shape, self.args))

    def _build_inputs(self, batch, t, discr_signal=None):
        # Assumes homogenous agents with flat observations.
        # Builds inputs for all agents, one timestep
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(
                0).expand(bs, -1, -1))
        if self.args.gail_obs_discr:
            inputs.append(discr_signal)
        inputs = th.cat(inputs, dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.gail_obs_discr:
            input_shape += 1


        return input_shape
