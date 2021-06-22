import torch
import fire
import ray
import chess.engine
import random
# import haiku as hk
# import jax
# import jax.numpy as np
# import numpy as onp
# import optax

# from jax import grad, vmap
# from models.role.policy_net import policy_fn
from play import play, play_contender, get_next_states, get_previous_network
from utils import set_seed, create_path
from models.role.agent import DoubleDQN
from pathlib import Path
from datetime import datetime
from models.role.replay_memory import ReplayMemory
from models.random_player import RandomPlayer
from models.role_player import Role
from utils import to_json, s2c
# from utils.moves import make_move
from torch.nn import functional as F


def optimize_network(network,
                     optimizer,
                     experience_replay,
                     num_updates,
                     batch_size,
                     discount):
    batch_loss = []
    for _ in range(num_updates):
        batch_loss.append(
            _optimize(network,
                      optimizer,
                      batch_size,
                      experience_replay,
                      discount)
        )

    return sum(batch_loss) / len(batch_loss)


def _optimize(network,
              optimizer,
              batch_size,
              experience_replay,
              discount):

    optimizer.zero_grad()
    experience = experience_replay.sample(batch_size)
    states_ = torch.stack([S for S, *_ in experience]).to(network.device)

    next_states_ = [S for *_, S, _ in experience]

    q_value = network.policy_net(states_).reshape((1, batch_size))

    q_target = torch.stack([
        network.target_net(S.to(network.device)).max(dim=0).values.detach()
        for S in next_states_
    ]).reshape((1, batch_size)).to(network.device)

    rewards = torch.stack([
        R for _, R, *_ in experience
    ]).reshape((1, batch_size)).to(network.device)

    is_terminal = torch.tensor([
        T for *_, T in experience
    ]).reshape((1, batch_size)).to(network.device)

    q_target = rewards + (1 - is_terminal) * discount * q_target

    loss = F.smooth_l1_loss(q_target, q_value, reduction="mean")

    loss.backward()
    optimizer.step()

    return loss.item()


def main(seed: int = 0,
         num_workers=1,
         num_gpus=1,
         max_num_games: int = 20000,
         max_moves: int = 256,
         discount: float = 0.9,
         experience_replay_size: int = 2,
         batch_size: int = 256,
         update_target_interval: int = 10,
         save_interval: int = 1000,
         num_updates: int = 5,
         target_update_interval: int = 1,
         expert_path: Path = Path('./stockfish'),
         save_dir: Path = Path('./runs'),
         optim: str = 'SGD',
         polyak_avg: bool = False,
         test_with_last_n: int = 3,
         **optim_config):

    set_seed(seed)

    ray.init(num_cpus=num_workers,
             num_gpus=num_gpus,
             include_dashboard=False)

    save_dir = save_dir / datetime.now().isoformat()
    create_path(save_dir)
    create_path(save_dir / 'train_history')
    create_path(save_dir / 'checkpoints')
    create_path(save_dir / 'games')

    num_features = 384
    experience_replay_size = experience_replay_size * max_moves * num_workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer_class = s2c(f"torch.optim.{optim}")

    experience_replay = ReplayMemory(experience_replay_size)

    # network = hk.without_apply_rng(hk.transform(policy_fn))

    # tx = optax.sgd(**optim_config)

    # key = jax.random.PRNGKey(seed)
    # params = network.init(key, np.ones((384,)))
    # opt_state = tx.init(params)

    network = DoubleDQN(
        device=device,
        eps_decay=0.999,
        num_features=num_features,
    )

    optimizer = optimizer_class(
        network.parameters(),
        **optim_config
    )

    losses = []
    num_games = 0
    target_update = 0

    ids = []

    residual = save_interval

    to_json(save_dir / 'configuration.json', {
        'seed': seed,
        'max_move': max_moves,
        'discount': discount,
        'experience_replay_size': experience_replay_size,
        'batch_size': batch_size,
        'num_updates': num_updates,
        'optim': optim,
        **optim_config,

    })

    while num_games < max_num_games:
        game_ids = []

        network.to('cpu')

        for _ in range(num_workers):
            game_ids.append(
                play.remote(network,
                            expert_path,
                            max_moves=max_moves)
            )

        for id_ in game_ids:
            replay = ray.get(id_)
            for experience in replay.memory:
                experience_replay.push(
                    *experience
                )

        network.to(device)

        residual -= num_workers
        num_games += num_workers
        target_update += 1

        if len(experience_replay) >= batch_size:
            loss = optimize_network(network,
                                    optimizer,
                                    experience_replay,
                                    num_updates,
                                    batch_size,
                                    discount)
            network.policy.update()
            if target_update >= target_update_interval:
                target_update = 0
                network.update_target_net(polyak_avg)
            losses.append(loss)
            print(f'[{num_games}/{max_num_games}] loss: {loss:.4f}')

            to_json(save_dir / "loss.json", losses)

            torch.save(network.policy_net.state_dict(),
                       save_dir / 'checkpoints' / f'model_{num_games}.pth')

            ids.append(num_games)

        if residual <= 0:
            residual = save_interval
            old_eps = network.policy.eps
            network.set_deterministic()

            role = Role(network, get_next_states)

            contender_ids = [random.choice(ids) for _ in range(test_with_last_n)]

            contenders = [
                get_previous_network(id_,
                                     num_features,
                                     save_dir / 'checkpoints',
                                     network)
                for id_ in contender_ids
            ]

            for contender in [
                    RandomPlayer(),
                    *contenders,
                    chess.engine.SimpleEngine.popen_uci(expert_path)
            ]:
                id = f"ROLEv{num_games}-vs-{contender.id['name']}"
                play_contender(role,
                               contender,
                               record_game=True,
                               game_id=id,
                               save_dir=save_dir / 'games')

            network.policy.eps = old_eps


if __name__ == '__main__':
    fire.Fire(main)
