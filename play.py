import chess
import chess.pgn
import chess.engine
import chess.svg

from models.random_player import RandomPlayer

stockfish = chess.engine.SimpleEngine.popen_uci("/home/danilo/.local/bin/stockfish")

A = RandomPlayer()
B = RandomPlayer()


board = chess.Board()
game = chess.pgn.Game()
node = None

moves = []

while not board.is_game_over() and len(moves) <= 200:
    print(stockfish.analyse(board, chess.engine.Limit(time=0.001))['score'].relative)
    result = stockfish.play(board, chess.engine.Limit(time=0.1))
    board.push(result.move)
    moves.append(result.move)

    result = stockfish.play(board, chess.engine.Limit(time=0.001))
    print(stockfish.analyse(board, chess.engine.Limit(time=0.1))['score'].relative)
    board.push(result.move)
    moves.append(result.move)

    # if node is None:
    #     node = game.add_variation(result.move)
    # else:
    #     node = node.add_variation(result.move)

game.add_line(moves)
A.quit()
B.quit()

pgn = open("game.pgn", "w", encoding="utf-8")
exporter = chess.pgn.FileExporter(pgn)

game.accept(exporter)

open('game.svg', 'w').write(chess.svg.board(board))
