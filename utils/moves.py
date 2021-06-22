# import numpy as np
# import jax
# import chess


# def to_1d_index(start, end, num_cols):
#     return start * num_cols + end


# def to_mask(legal_moves):
#     mask = np.zeros((4096,), dtype=np.bool8)

#     legal_idxs = [to_1d_index(m.from_square, m.to_square, 64) for m in legal_moves]
#     mask[legal_idxs] = np.bool8(True)
#     indices = (1-mask).nonzero()[0]
#     return mask, indices


# def make_move(x, network, params, board):
#     log_prob, value = network.apply(params, x)

#     _, mask = to_mask(board.legal_moves)

#     move_idx = jax.ops.index_update(log_prob, mask, -100).argmax()

#     from_square = move_idx // 64
#     to_square = move_idx % 64

#     return log_prob, move_idx, chess.Move(from_square, to_square)
