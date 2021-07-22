import chess
import chess.engine
import jax.numpy as np


class Role:

    def __init__(self,
                 network,
                 params,
                 get_next_states,
                 id="Role"):
        self.id = {
            'name': id
        }
        self.network = network
        self.params = params
        self.get_next_states = get_next_states

    def play(self, board: chess.Board, limit=None) -> chess.engine.PlayResult:

        legal_moves = list(board.legal_moves)
        next_states = np.stack(
            self.get_next_states(board=board)
        )

        move_idx = self.network.apply(self.params, next_states).argmax()

        return chess.engine.PlayResult(move=legal_moves[move_idx],
                                       ponder=None)

    def quit(self):
        pass
