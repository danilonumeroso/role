import torch
import fire
import ray
import chess.engine

from play import play, play_contender, get_next_states
from utils import set_seed, create_path
from models.role.agent import DoubleDQN
from pathlib import Path
from datetime import datetime
from models.role.replay_memory import ReplayMemory
from models.random_player import RandomPlayer
from models.role_player import Role
from utils import to_json, s2c


def main(seed: int = 0,
         num_cpus=1,
         num_gpus=1,
         max_num_games: int = 20000,
         max_moves: int = 256,
         discount: float = 0.9,
         experience_replay_size: int = 2,
         batch_size: int = 256,
         update_target_interval: int = 10,
         save_interval: int = 1000,
         num_updates: int = 5,
         expert_path: Path = Path('./stockfish'),
         save_dir: Path = Path('./runs'),
         optim: str = 'SGD',
         **optim_config):

    set_seed(seed)

    ray.init(num_cpus=num_cpus,
             num_gpus=num_gpus,
             include_dashboard=False)

    save_dir = save_dir / datetime.now().isoformat()
    create_path(save_dir)
    create_path(save_dir / 'train_history')

    num_features = 384
    experience_replay_size = experience_replay_size * max_moves * num_cpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer_class = s2c(f"torch.optim.{optim}")

    experience_replay = ReplayMemory(experience_replay_size)
    network = DoubleDQN(
        device=device,
        eps_decay=0.999,
        num_features=num_features,
        batch_size=batch_size,
        discount=discount,
        optimizer_class=optimizer_class,
        **optim_config
    )

    losses = []
    num_games = 0

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
        for _ in range(num_cpus):
            game_ids.append(
                play.remote(network,
                            expert_path,
                            max_moves=max_moves)
            )

        for id_ in game_ids:
            replay, history = ray.get(id_)
            for experience in replay.memory:
                experience_replay.push(
                    *experience
                )

        to_json(save_dir / "train_history" / f"{num_games}.json", history)

        network.to(device)

        residual -= num_cpus
        num_games += num_cpus

        if len(experience_replay) >= batch_size:
            loss = network.optimize(experience_replay, num_updates)
            network.update_target_net()
            losses.append(loss)
            print(f'[{num_games}/{max_num_games}] loss: {loss:.4f}')

            to_json(save_dir / "loss.json", losses)
            torch.save(network.policy_net.state_dict(),
                       save_dir / f'model_{num_games}.pth')

        if residual <= 0:
            residual = save_interval
            old_eps = network.policy.eps
            network.policy.eps = 0.0

            role = Role(network, get_next_states)
            for contender in [
                    RandomPlayer(),
                    Role(network, get_next_states),
                    chess.engine.SimpleEngine.popen_uci(expert_path)
            ]:
                id = f"role-vs-{contender.id['name']}"
                play_contender(role,
                               contender,
                               record_game=True,
                               game_id=id,
                               save_dir=save_dir)

            network.policy.eps = old_eps


if __name__ == '__main__':
    fire.Fire(main)
