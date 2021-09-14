import random
import fire
import chess
import haiku as hk

from models.role.dqn import policy_fn
from utils import load
from play import play_contender, get_next_states
from pathlib import Path
from models.human_player import HumanPlayer
from models.role_player import Role


def main(model_path: Path,
         expert_path: Path = Path("./stockfish"),
         ):

    network = hk.without_apply_rng(hk.transform(policy_fn))

    role = Role(network,
                load(model_path),
                get_next_states,
                id="Role")

    contender = HumanPlayer()
    stockfish = chess.engine.SimpleEngine.popen_uci(expert_path, )

    white, black = (network, contender) if random.uniform(0, 1) >= 0.5 else (contender, network)

    play_contender(role,
                   contender,
                   save_dir=Path('.'),
                   game_id='test',
                   verbose=True,
                   time_per_move=1e-10)


if __name__ == "__main__":
    fire.Fire(main)
