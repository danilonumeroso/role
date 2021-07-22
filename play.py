import chess
import chess.pgn
import chess.engine
import chess.svg
import ray
import random
import jax

import jax.numpy as np

from models.role.replay_memory import ReplayMemory
from utils.features import board_to_feature
from utils.data import normalize
from models.role.dqn import eps_greedy

MATE_SCORE = 3000
SCORE_REDUCTION = 100
CLAMP_VALUE = 10
TIME_PER_MOVE = 1e-9


def get_next_states(board):
    next_states = []

    for move in board.legal_moves:
        board.push(move)

        next_states.append(
            np.array(board_to_feature(board))
        )

        board.pop()

    next_states = np.array(next_states)

    return next_states


def get_reward(board,
               expert,
               white_turn,
               premove_analysis):

    r = expert.analyse(
        board,
        chess.engine.Limit(time=TIME_PER_MOVE)
    )['score'].pov(white_turn)

    r = r.score(mate_score=MATE_SCORE) / SCORE_REDUCTION
    r = r - (premove_analysis.score(mate_score=MATE_SCORE) / SCORE_REDUCTION)
    r = min(CLAMP_VALUE, max(r, -CLAMP_VALUE))

    range_ = (-1, 0)

    return normalize(r, x_min=-CLAMP_VALUE, x_max=CLAMP_VALUE, range_=range_)


@ray.remote(num_cpus=1, num_gpus=0)
def play(network,
         params,
         rng,
         epsilon,
         expert_path,
         expert_ratio=.1,
         max_moves=256,):

    expert = chess.engine.SimpleEngine.popen_uci(expert_path)

    board = chess.Board()

    replay_memory = ReplayMemory(max_moves)

    for i in range(max_moves):
        white_turn = board.turn

        legal_moves = list(board.legal_moves)

        premove_analysis = expert.analyse(
            board,
            chess.engine.Limit(time=TIME_PER_MOVE)
        )['score'].pov(white_turn)

        next_states = np.stack(
            get_next_states(board=board)
        )

        expert_move = jax.random.uniform(next(rng)) < expert_ratio

        if expert_move:
            m = expert.play(board, chess.engine.Limit(time=TIME_PER_MOVE)).move
        else:
            move_idx, _ = eps_greedy(rng, network, params, next_states, epsilon)
            m = legal_moves[move_idx]

        board.push(m)

        score = expert.analyse(
            board,
            chess.engine.Limit(time=TIME_PER_MOVE)
        )['score'].pov(white_turn)

        if expert_move:  # or m == best_move:
            reward = 2.0 if score.is_mate() else 1.0
        else:
            reward = get_reward(board,
                                expert,
                                white_turn,
                                premove_analysis)

        is_terminal = board.is_game_over()

        replay_memory.push(
            next_states[move_idx],
            np.array(reward),
            next_states,
            int(is_terminal)
        )

        if is_terminal:
            break

    expert.quit()

    return replay_memory


def play_contender(network,
                   contender,
                   max_moves=512,
                   time_per_move=0.001,
                   record_game=True,
                   save_dir=None,
                   game_id=None,
                   verbose=False,
                   callback=None):

    board = chess.Board()
    game = chess.pgn.Game()
    moves = []

    white, black = (network, contender) if random.uniform(0, 1) >= 0.5 else (contender, network)

    game.headers["Event"] = f"Game {game_id}"
    game.headers["Site"] = "Virtual"
    game.headers["White"] = white.id['name']
    game.headers["Black"] = black.id['name']

    for i in range(max_moves//2):
        player = white if board.turn else black

        m = player.play(board, chess.engine.Limit(time=time_per_move)).move

        if verbose:
            print(board.san(m))

        board.san_and_push(m)
        moves.append(m)

        if callback:
            callback(board)

        if board.is_game_over():
            break

    if record_game:
        game.add_line(moves)

        with open(save_dir / f"game_{game_id}.pgn", "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

    contender.quit()
