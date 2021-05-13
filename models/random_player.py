import random
import chess
import chess.engine

class RandomPlayer:

    def __init__(self):
        self.id = {
            'name': 'RandomPlayer'
        }

    def play(self, board: chess.Board, limit=None) -> chess.engine.PlayResult:

        legal_moves = list(board.legal_moves)
        move = random.choice(legal_moves)

        return chess.engine.PlayResult(move=move, ponder=None)

    def quit(self):
        pass
