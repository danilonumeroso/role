import torch
from torch.nn import functional as F
from models.role.dqn import DQN
from models.policy import DecayingEpsilonGreedy


class DoubleDQN:
    def __init__(self,
                 device,
                 num_features,
                 batch_size,
                 discount,
                 eps_decay,
                 optimizer_class,
                 **optimizer_config,
                 ):

        self.device = device
        self.num_input = num_features
        self.policy = DecayingEpsilonGreedy(eps_decay)
        self.batch_size = batch_size
        self.discount = discount

        self.policy_net, self.target_net = (
            DQN(num_features, 1).to(self.device),
            DQN(num_features, 1).to(self.device)
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = optimizer_class(
            self.policy_net.parameters(),
            **optimizer_config
        )

    def to(self, device):
        self.device = device

        self.policy_net, self.target_net = (
            self.policy_net.to(device),
            self.target_net.to(device)
        )

    def set_deterministic(self):
        self.policy.eps = 0

    def take_action(self, states):
        return self.policy.take(
            self.policy_net,
            states.to(self.device)
        ).cpu().detach().numpy()

    def optimize(self, experience_replay, num_updates=1):

        batch_loss = []
        for _ in range(num_updates):
            batch_loss.append(
                self._optimize(experience_replay)
            )

        self.policy.update()
        return sum(batch_loss) / len(batch_loss)

    def _optimize(self, experience_replay):

        self.optimizer.zero_grad()
        experience = experience_replay.sample(self.batch_size)
        states_ = torch.stack([S for S, *_ in experience]).to(self.device)

        next_states_ = [S for *_, S, _ in experience]

        q = self.policy_net(states_).reshape((1, self.batch_size))

        q_target = torch.stack([
            self.target_net(S.to(self.device)).max(dim=0).values.detach()
            for S in next_states_
        ]).reshape((1, self.batch_size)).to(self.device)

        rewards = torch.stack([
            R for _, R, *_ in experience
        ]).reshape((1, self.batch_size)).to(self.device)

        is_terminal = torch.tensor([
            T for *_, T in experience
        ]).reshape((1, self.batch_size)).to(self.device)

        q_target = rewards + self.discount * (1 - is_terminal) * q_target

        loss = F.smooth_l1_loss(q, q_target, reduction="mean")

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
