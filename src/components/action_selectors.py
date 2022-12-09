import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule, FlatSchedule


REGISTRY = {}


class MultinomialActionSelector():
    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, t_ep, test_mode=False):
        q_values = agent_inputs.clone()

        if self.args.algo_name == "isql":
            values = self.args.alpha * th.logsumexp(q_values / self.args.alpha, dim=2, keepdims=True)
            logits = 1. / self.args.alpha * (q_values - values)
            # print("AVAIL ACTIONS ARE ", avail_actions)
            logits[avail_actions == 0.0] = -1e10
            # print("LOGITS ARE ", logits)
            exp_logits = th.exp(logits) # shape: (1, n_agents, n_actions)

            # if all actions for agent have prob 0, add prob mass to no-op
            # summed_logits = th.sum(exp_logits, dim=2) # .repeat(1, 1, n_actions)
            # noop_logits = exp_logits[:, :, 0]
            # ones = th.ones_like(noop_logits)
            # corrected_noop_logits = th.where(summed_logits != 0., noop_logits, ones)
            # exp_logits[:, :, 0] = corrected_noop_logits

            picked_actions = Categorical(exp_logits).sample().long()

        else:        
            q_values[avail_actions == 0.0] = 0.0
            if test_mode and self.test_greedy:
                picked_actions = q_values.max(dim=2)[1]
            else:
                picked_actions = Categorical(q_values).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = 0.0
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector