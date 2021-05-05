import torch
import fire
import numpy as np

from utils import set_seed
from torch.nn import functional as F
from models.role.agent import Agent


def main(num_episodes: int = 5000,
         lr: float = 1e-4,
         polyak: float = 0.995,
         gamma: float = 0.95,
         discount: float = 0.9,
         replay_buffer_size: int = 10000,
         batch_size: int = 1,
         update_interval: int = 1,
         seed: int = 0):

    set_seed(seed)

    num_input = 353
    num_output = 1

    device = torch.device("cpu")
    agent = Agent(num_input, num_output, device, lr, replay_buffer_size)

    eps = 0.05
    batch_losses = []
    episode = 0
    it = 0
    environment = None

    while episode < num_episodes:
        valid_actions = list(environment.get_valid_actions())

        # observations = np.vstack(
        #     [
        #         np.append(action_encoder(action), steps_left)
        #         for action in valid_actions
        #     ]
        # )

        observations = None

        observations = torch.as_tensor(observations).float()
        a = agent.action_step(observations, eps)
        action = valid_actions[a]

        result = environment.step(action)

        # action_embedding = np.append(
        #     action_encoder(action),
        #     steps_left
        # )

        _, out, done = result

        agent.replay_buffer.push(
            # torch.as_tensor(action_embedding).float(),
            # torch.as_tensor(out['reward']).float(),
            # torch.as_tensor(action_embeddings).float(),
            float(result.terminated)
        )

        if it % update_interval == 0 and len(agent.replay_buffer) >= batch_size:
            loss = agent.train_step(
                batch_size,
                gamma,
                polyak
            )
            loss = loss.item()
            batch_losses.append(loss)

        it += 1

        if done:
            episode += 1

            # print(f'Episode {episode}> Reward = {out["reward"]:.4f} (pred: {out["reward_pred"]:.4f}, sim: {out["reward_sim"]:.4f})')

            # eps *= 0.9995


if __name__ == '__main__':
    fire.Fire(main)
