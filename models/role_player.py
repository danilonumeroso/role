import chess
import chess.engine
import torch

MATE_SCORE = 1000.0


def _evaluate(board, network, get_next_states):
    """Score the current position using the network (current player's POV)."""
    next_states = get_next_states(board)
    if not next_states:
        return 0.0
    with torch.no_grad():
        values = network.policy_net(torch.stack(next_states).to(network.device))
    return values.max().item()


def _negamax(board, depth, alpha, beta, network, get_next_states):
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_game_over():   # stalemate, draw by repetition, etc.
        return 0.0
    if depth == 0:
        return _evaluate(board, network, get_next_states)

    best = -float('inf')
    for move in board.legal_moves:
        board.push(move)
        score = -_negamax(board, depth - 1, -beta, -alpha, network, get_next_states)
        board.pop()
        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break
    return best


class Role:

    def __init__(self, network, get_next_states, depth=0):
        self.id = {'name': 'Role'}
        self.network = network
        self.get_next_states = get_next_states
        self.depth = depth

    def play(self, board: chess.Board, limit=None) -> chess.engine.PlayResult:
        legal_moves = list(board.legal_moves)

        if self.depth == 0:
            next_states = torch.stack(self.get_next_states(board=board))
            move_idx = self.network.take_action(next_states)
            return chess.engine.PlayResult(move=legal_moves[move_idx], ponder=None)

        best_move, best_score = None, -float('inf')
        alpha = -float('inf')

        for move in legal_moves:
            board.push(move)
            score = -_negamax(board, self.depth - 1, -float('inf'), -alpha,
                              self.network, self.get_next_states)
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)

        return chess.engine.PlayResult(move=best_move, ponder=None)

    def quit(self):
        pass
