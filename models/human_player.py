import random
import chess
import chess.engine


class HumanPlayer:

    def __init__(self):
        self.id = {
            'name': 'Human'
        }

    def play(self, board: chess.Board, limit=None) -> chess.engine.PlayResult:

        while True:
            try:
                move = input("Move: ")
                move = chess.Move.from_uci(move)
                if move not in board.legal_moves:
                    raise Exception("Aaaaaaaah")
                break
            except KeyboardInterrupt:
                raise
            except:
                pass


        return chess.engine.PlayResult(move=move, ponder=None)

    def quit(self):
        pass
