import chess
import chess.pgn
import chess.engine
import chess.svg
import ray
import random
import torch

from models.role.replay_memory import ReplayMemory
from utils.features import board_to_feature

MATE_SCORE = 3000
SCORE_REDUCTION = 100


def get_next_states(board):
    next_states = []
    for move in board.legal_moves:
        board.push(move)

        next_states.append(
            torch.as_tensor(board_to_feature(board)).float()
        )

        board.pop()

    return next_states


def get_reward(board,
               expert,
               previous_reward,
               time_per_move):
    r = expert.analyse(
        board,
        chess.engine.Limit(time=time_per_move)
    )['score'].pov(board.turn)
    r = r.score(mate_score=MATE_SCORE) / SCORE_REDUCTION
    r = r - previous_reward

    return r


def move(board,
         legal_moves,
         next_states,
         network,
         expert,
         expert_ratio,
         time_per_move):
    expert_move = random.uniform(0, 1) <= expert_ratio

    if expert_move:
        move = expert.play(board, chess.engine.Limit(time=time_per_move)).move
        return legal_moves.index(move), expert_move
    else:
        move_idx = network.take_action(next_states)
        return move_idx, expert_move


@ray.remote(num_cpus=1, num_gpus=0)
def play(network,
         expert_path,
         expert_ratio=0.5,
         time_per_move=0.001,
         max_moves=256,
         record_game=False,
         game_id=None,
         save_dir=None):

    expert = chess.engine.SimpleEngine.popen_uci(expert_path)

    board = chess.Board()
    game = chess.pgn.Game()
    moves = []
    history = []
    white_reward = 0
    black_reward = 0

    replay_memory = ReplayMemory(max_moves)

    for i in range(max_moves):
        white_turn = board.turn

        legal_moves = list(board.legal_moves)

        next_states = torch.stack(
            get_next_states(board=board)
        )

        move_idx, expert_move = move(board,
                                     legal_moves,
                                     next_states,
                                     network,
                                     expert,
                                     expert_ratio,
                                     time_per_move)

        m = legal_moves[move_idx]
        board.push(m)

        reward = get_reward(board,
                            expert,
                            white_reward if white_turn else black_reward,
                            time_per_move=time_per_move)

        if white_turn:
            white_reward = reward
        else:
            black_reward = reward

        if record_game:
            moves.append(m)

        is_terminal = board.is_game_over()

        replay_memory.push(
            next_states[move_idx],
            torch.as_tensor(reward).float(),
            next_states,
            int(is_terminal)
        )

        history.append({
            'R': reward,
            'action': str(m),
            'expert_move': expert_move,
            'random_action': network.policy.random_action,
            'color': 'white' if white_turn else 'black'
        })

        if is_terminal:
            break

    if record_game:
        game.add_line(moves)
        with open(save_dir / f"game_{game_id}.pgn", "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

    expert.quit()

    return replay_memory, history


def play_contender(network,
                   contender,
                   max_moves=512,
                   record_game=True,
                   save_dir=None,
                   game_id=None):

    board = chess.Board()
    game = chess.pgn.Game()
    moves = []

    white, black = network, contender if random.uniform(0,1) >= 0.5 else contender, network

    game.headers["Event"] = f"Game {game_id}"
    game.headers["Site"] = "Virtual"
    game.headers["White"] = white.id['name']
    game.headers["Black"] = black.id['name']

    for i in range(max_moves//2):

        m = white.play(board).move
        board.push(m)
        moves.append(m)

        m = black.play(board).move
        board.push(m)
        moves.append(m)

        if board.is_game_over():
            break

    if record_game:
        game.add_line(moves)

        with open(save_dir / f"game_{game_id}.pgn", "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)

    contender.quit()
