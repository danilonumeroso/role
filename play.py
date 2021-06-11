import chess
import chess.pgn
import chess.engine
import chess.svg
import ray
import random
import torch

import jax.numpy as np

from models.role.replay_memory import ReplayMemory, Transitions
from utils.features import board_to_feature
from utils.data import normalize
from utils.moves import make_move, to_1d_index

MATE_SCORE = 3000
SCORE_REDUCTION = 100
CLAMP_VALUE = 10


def get_next_states(board):
    next_states = []
    for move in board.legal_moves:
        board.push(move)

        next_states.append(
            torch.as_tensor(board_to_feature(board)).float()
        )

        board.pop()

    return next_states


def get_previous_network(id,
                         num_features,
                         checkpoint_dir,
                         current_network,
                         device=torch.device('cpu')):
    try:
        contender_network = DoubleDQN(
            device='cpu',
            eps_start=0.,
            num_features=num_features,
        )

        contender_network.set_network(
            torch.load(
                checkpoint_dir / f"model_{id}.pth",
                map_location=device
            )
        )

        role_contender = Role(contender_network,
                              get_next_states)

        role_contender.id = {'name': f'ROLEv{id}'}

        return role_contender
    except FileNotFoundError:

        role_contender = Role(current_network,
                              get_next_states)
        role_contender.id = {'name': 'ROLE='}

    return role_contender


# def get_reward(board,
#                expert,
#                white_turn,
#                premove_analysis,
#                time_per_move):

#     r = expert.analyse(
#         board,
#         chess.engine.Limit(time=time_per_move)
#     )['score'].pov(white_turn)

#     r = r.score(mate_score=MATE_SCORE) / SCORE_REDUCTION
#     r = r - premove_analysis

#     if r >= -0.01:
#         return 0.5

#     r = min(CLAMP_VALUE, max(r, -CLAMP_VALUE))

#     range_ = (-1, 1)

#     return normalize(r, x_min=-CLAMP_VALUE, x_max=CLAMP_VALUE, range_=range_)


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
        return torch.tensor(random.randint(0, next_states.size(0)-1)), expert_move
        # move_idx = network.take_action(next_states)
        # return move_idx, expert_move


@ray.remote(num_cpus=1, num_gpus=0)
def play(network,
         params,
         expert_path,
         expert_ratio=.6,
         time_per_move=0.001,
         max_moves=256,
         record_game=False,
         game_id=None,
         save_dir=None):

    expert = chess.engine.SimpleEngine.popen_uci(expert_path)

    board = chess.Board()
    # game = chess.pgn.Game()
    # moves = []
    # history = []

    # replay_memory = ReplayMemory(max_moves)

    white = []
    black = []

    for i in range(max_moves):
        white_turn = board.turn

        # next_states = torch.stack(
        #     get_next_states(board=board)
        # )

        # best_move = expert.play(board,
        # chess.engine.Limit(time=time_per_move)).move

        expert_move = random.uniform(0, 1) <= expert_ratio

        state = np.array(board_to_feature(board))
        # if expert_move:
        #     move = expert.play(board, chess.engine.Limit(time=time_per_move))
        #     assert move.move is not None, move

        #     idx = to_1d_index(move.from_square, move.to_square, 64)
        # else:

        _, idx, move = make_move(state, network, params, board)
        idx = idx.item()
        if i < 3:
            print(move)
        # move_idx, expert_move = move(board,
        #                              legal_moves,
        #                              next_states,
        #                              network,
        #                              expert,
        #                              expert_ratio,
        #                              time_per_move)

        # m = legal_moves[move_idx]
        # board.push(m)

        board.push(move)

        # score = expert.analyse(
        #     board,
        #     chess.engine.Limit(time=time_per_move)
        # )['score'].pov(white_turn)

        # if expert_move:  # or m == best_move:
        # reward = 1.0 if score.is_mate() else 0.8
        # else:
        # reward = get_reward(board,
        #                     expert,
        #                     white_turn,
        #                     premove_analysis,
        #                     time_per_move=time_per_move)
        # reward = -1.0

        # assert reward >= -1 and reward <= 1

        (white if white_turn else black).append(
            Transitions(state, idx, None, 0)
        )

        if board.is_game_over() or i == max_moves - 1:
            tw, tb = white.pop(), black.pop()
            outcome = board.outcome()
            print(outcome)
            white.append(
                Transitions(state=tw.state,
                            action=tw.action,
                            state_next=tw.state_next,
                            reward=0.5 if not outcome and not outcome.winner else (
                                1 if outcome.winner else 0
                            ))
            )
            black.append(
                Transitions(state=tb.state,
                            action=tb.action,
                            state_next=tb.state_next,
                            reward=0.5 if not outcome and not outcome.winner else (
                                0 if outcome.winner else 1
                            ))
            )


            if outcome.winner is not None:
                print("Winner", chess.Color[outcome.winner])
            break

        # replay_memory.push(
        #     next_states[move_idx],
        #     torch.as_tensor(reward).float(),
        #     next_states,
        #     int(is_terminal)
        # )

        # history.append({
        #     'R': reward,
        #     'action': str(m),
        #     'expert_move': expert_move,
        #     'random_action': reward < 0,  # network.policy.random_action and not expert_move,
        #     'color': 'white' if white_turn else 'black'
        # })

    # expert.quit()

    return white, black  # , history


def play_contender(network,
                   contender,
                   max_moves=512,
                   time_per_move=0.001,
                   record_game=True,
                   save_dir=None,
                   game_id=None,
                   verbose=False):

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

        if verbose:
            print(list(board.legal_moves))

        m = player.play(board, chess.engine.Limit(time=time_per_move)).move

        if verbose:
            print(m)

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
