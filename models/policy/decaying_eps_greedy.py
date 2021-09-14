import random
# import torch
from models.policy.policy import BasePolicy


class DecayingEpsilonGreedy(BasePolicy):
    def __init__(self,
                 eps_decay,
                 eps_start=1.,
                 eps_end=0.):
        super(DecayingEpsilonGreedy, self).__init__()

        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.random_action = None
        self._eps_history = [eps_start]

    def take(self, policy_net, states: torch.Tensor):
        self.random_action = random.uniform(0, 1) < self.eps

        if self.random_action:
            return torch.tensor(random.randint(0, states.size(0)-1))
        else:
            q_values = policy_net(states)
            return torch.argmax(q_values)

    def update(self):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        self._eps_history.append(self.eps)

        return self.eps

    def random_action(self):
        return self.random_action
