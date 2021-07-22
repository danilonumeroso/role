import fire
import ray
import chess.engine
import random
import haiku as hk
import jax.numpy as np
import optax

from models.role.dqn import policy_fn, train_fns, update_target_net
from play import play, play_contender, get_next_states
from utils import set_seed, create_path, dump, load
from pathlib import Path
from datetime import datetime
from models.role.replay_memory import ReplayMemory
from models.random_player import RandomPlayer
from models.role_player import Role
from haiku.data_structures import to_immutable_dict as hk_clone
from utils import to_json


def optimize_net(optimize,
                 loss_fn,
                 num_updates,
                 replay,
                 w_policy,
                 w_target,
                 opt_state
                 ):
    batch_loss = []
    for _ in range(num_updates):
        params, opt_state, loss = optimize(replay,
                                           w_policy,
                                           w_target,
                                           opt_state)

        batch_loss.append(loss)

    return params, opt_state, sum(batch_loss) / len(batch_loss)


def main(seed: int = 0,
         num_workers: int = 1,
         num_gpus: int = 1,
         max_num_games: int = 20000,
         max_moves: int = 256,
         discount: float = 0.9,
         eps_decay: float = 0.999,
         experience_replay_size: int = 2,
         batch_size: int = 256,
         save_interval: int = 1000,
         num_updates: int = 5,
         expert_path: Path = Path('./stockfish'),
         save_dir: Path = Path('./runs'),
         optim: str = 'SGD',
         polyak_avg: bool = False,
         num_contenders: int = 3,
         **optim_config):

    set_seed(seed)

    ray.init(num_cpus=num_workers,
             num_gpus=num_gpus,
             include_dashboard=False)

    save_dir = save_dir / datetime.utcnow().isoformat()[:-7]
    create_path(save_dir)
    create_path(save_dir / 'train_history')
    create_path(save_dir / 'checkpoints')
    create_path(save_dir / 'games')

    num_features = 384
    experience_replay_size = experience_replay_size * max_moves * num_workers
    experience_replay = ReplayMemory(experience_replay_size)

    network = hk.without_apply_rng(hk.transform(policy_fn))

    rng = hk.PRNGSequence(seed)

    w_policy = network.init(next(rng), np.empty((num_features,)))
    w_target = hk_clone(w_policy)

    opt = optax.sgd(**optim_config)
    opt_state = opt.init(w_policy)

    loss_fn, optimize = train_fns(network=network,
                                  opt=opt,
                                  batch_size=batch_size,
                                  discount=discount)

    losses = []
    num_games = 0
    ids = []
    residual = save_interval
    eps = 1.

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

        for i in range(num_workers):
            base_rng = next(rng)
            game_ids.append(
                play.remote(network,
                            w_policy,
                            hk.PRNGSequence(base_rng + i),
                            eps,
                            expert_path,
                            max_moves=max_moves)
            )

        for id_ in game_ids:
            replay = ray.get(id_)
            for experience in replay.memory:
                experience_replay.push(
                    *experience
                )

        residual -= num_workers
        num_games += num_workers

        if len(experience_replay) >= batch_size:

            params, opt_state, loss = optimize_net(optimize,
                                                   loss_fn,
                                                   num_updates,
                                                   replay,
                                                   w_policy,
                                                   w_target,
                                                   opt_state)

            update_target_net(w_policy, w_target)
            eps = eps * eps_decay
            losses.append(loss)

            print(f'[{num_games}/{max_num_games}] loss: {loss:.4f}')
            np.save(f"{save_dir}/loss", losses)
            dump(w_policy, save_dir / 'checkpoints' / f'model_{num_games}.ckpt')

            ids.append(num_games)

        if residual <= 0:
            residual = save_interval

            role = Role(network,
                        w_policy,
                        get_next_states,
                        id=f"Role {num_games}")
            try:
                contenders = [
                    Role(network,
                         load(save_dir / 'checkpoints' / f'model_{id_}.ckpt'),
                         get_next_states,
                         id=f"Role {id_}")
                    for id_ in [random.choice(ids) for _ in range(num_contenders)]
                ]
            except FileNotFoundError:
                contenders = [role]
                pass

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


if __name__ == '__main__':
    fire.Fire(main)
