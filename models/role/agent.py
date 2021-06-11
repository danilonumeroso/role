import torch

from models.role.dqn import DQN
from models.policy import DecayingEpsilonGreedy
from torch.nn import functional as F


class DoubleDQN:
    def __init__(self,
                 device,
                 num_features,
                 eps_decay=0.99,
                 eps_start=1.,
                 eps_end=0.,
                 polyak_factor=0.99
                 ):

        self.device = device
        self.num_input = num_features
        self.polyak_factor = polyak_factor
        self.policy = DecayingEpsilonGreedy(
            eps_decay,
            eps_start,
            eps_end,
        )

        self.policy_net, self.target_net = (
            DQN(num_features, 1).to(self.device),
            DQN(num_features, 1).to(self.device)
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        for p in self.target_net.parameters():
            p.requires_grad = False

    def parameters(self):
        return self.policy_net.parameters()

    def set_network(self, state_dict):
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    def to(self, device):
        self.device = device

        self.policy_net, self.target_net = (
            self.policy_net.to(device),
            self.target_net.to(device)
        )

    def set_deterministic(self):
        self.policy.eps = 0
        self.policy.eps_end = 0

    def take_action(self, states):
        return self.policy.take(
            self.policy_net,
            states.to(self.device)
        ).cpu().detach().numpy()

    def update_target_net(self, polyak_avg=False):

        if polyak_avg:
            with torch.no_grad():
                for policy_param, target_param in zip(
                        self.policy_net.parameters(),
                        self.target_net.parameters()
                ):
                    target_param.data.mul_(self.polyak_factor)
                    target_param.data.add_((1 - self.polyak_factor) * policy_param.data)

                p_params = []
                t_params = []

                for policy_param, target_param in zip(
                        self.policy_net.parameters(),
                        self.target_net.parameters()
                ):

                    p_params.append(policy_param.flatten())
                    t_params.append(target_param.flatten())

                p_params = torch.cat(p_params)
                t_params = torch.cat(t_params)

                print(f"Parameters distance: {F.l1_loss(p_params, t_params).item()}")

            return

        self.target_net.load_state_dict(self.policy_net.state_dict())
