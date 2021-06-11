from train import get_previous_network
from play import get_next_states, play_contender
from pathlib import Path
from models.human_player import HumanPlayer
import random
import chess

checkpoint_dir = Path('.')

network = get_previous_network('141952', 384, checkpoint_dir, None)
contender = HumanPlayer()

board = chess.Board()
game = chess.pgn.Game()
moves = []

expert = chess.engine.SimpleEngine.popen_uci('/home/danilo/.local/bin/stockfish')

white, black = (network, contender) if random.uniform(0, 1) >= 0.5 else (contender, network)

for i in range(256//2):

    player = white if board.turn else black
    print(expert.analyse(
        board,
        chess.engine.Limit(time=0.001)
    )['score'].pov(True))
    m = player.play(board, chess.engine.Limit(time=1)).move

    board.push(m)
    moves.append(m)

    print(m)

    if board.is_game_over():
        break
