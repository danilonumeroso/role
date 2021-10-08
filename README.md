# ROLE

Hello there! ROLE is an artificial, reinforcement learning based chess player.

It is trained by a mixture of self-play (Deep Q Learning + Experience Replay) and Imitation Learning on the 
[Stockfish](https://github.com/official-stockfish/Stockfish) engine, in which expert moves are directly input to the
replay memory.

Q Learning is a model-free RL algorithm. This means that ROLE plays without any look-ahead mechanisms, 
unlike Monte-Carlo Tree Search based methods, which is not a desirable property for playing chess.

However, this project was built to have fun rather than deploy yet another chess engine with superhuman performance :)

# Have fun
[Play ROLE on Lichess!](https://lichess.org/@/rolechess) (you'll need a Lichess account)
#### Available mode:
- Bullet
- Blitz


# Acknowledgments
- [DeepPepper](https://github.com/saikrishna-1996/deep_pepper_chess), from which I borrowed the state representation.
- [Stockfish](https://github.com/official-stockfish/Stockfish).
