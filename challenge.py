import random
import fire
import chess

from train import get_previous_network
from play import play_contender
from pathlib import Path
from models.human_player import HumanPlayer


def main(net: str,
         checkpoint_dir: Path = Path("./ckpt"),
         expert_path: Path = Path("./stockfish"),
         ):

    network = get_previous_network(net, 384, checkpoint_dir, None)
    contender = HumanPlayer()

    stockfish = chess.engine.SimpleEngine.popen_uci(expert_path, )

    white, black = (network, contender) if random.uniform(0, 1) >= 0.5 else (contender, network)

    play_contender(stockfish,
                   contender,
                   save_dir=Path('.'),
                   game_id='test',
                   verbose=True,
                   time_per_move=1e-10)


if __name__ == "__main__":
    fire.Fire(main)
