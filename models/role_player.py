import chess
import chess.engine
import torch


class Role:

    def __init__(self,
                 network,
                 get_next_states):
        self.id = {
            'name': 'Role'
        }
        self.network = network
        self.get_next_states = get_next_states

    def play(self, board: chess.Board, limit=None) -> chess.engine.PlayResult:

        legal_moves = list(board.legal_moves)
        next_states = torch.stack(
            self.get_next_states(board=board)
        )

        move_idx = self.network.take_action(next_states)

        return chess.engine.PlayResult(move=legal_moves[move_idx],
                                       ponder=None)

    def quit(self):
        pass
