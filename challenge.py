import random
import fire
import chess
import torch

from play import play_contender, get_previous_network
from pathlib import Path
from models.human_player import HumanPlayer
from pytorch2keras.converter import pytorch_to_keras


def main(net: str,
         checkpoint_dir: Path = Path("./checkpoints"),
         expert_path: Path = Path("./stockfish"),
         ):

    network = get_previous_network(net, 384, checkpoint_dir, None)
    input(type(network.network.policy_net))
    net = network.network.policy_net

    keras_model = pytorch_to_keras(net, torch.randn((1, 384)), input_shapes=[(None, 384)], name_policy='short', verbose=True)
    keras_model.save('keras.h5')

    contender = HumanPlayer()

    stockfish = chess.engine.SimpleEngine.popen_uci(expert_path, )

    white, black = (network, contender) if random.uniform(0, 1) >= 0.5 else (contender, network)

    play_contender(network,
                   contender,
                   save_dir=Path('.'),
                   game_id='test',
                   verbose=True,
                   time_per_move=1e-10)


if __name__ == "__main__":
    fire.Fire(main)
