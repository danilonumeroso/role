import torch
import fire
import chess
import chess.pgn

from utils import set_seed, create_path
from utils.features import board_to_feature
from models.role.agent import Agent
from pathlib import Path
from datetime import datetime


def main(num_episodes: int = 20000,
         len_episode: int = 256,
         lr: float = 1e-3,
         momentum: float = 0.9,
         polyak: float = 0.995,
         gamma: float = 0.95,
         discount: float = 0.9,
         replay_buffer_size: int = 10000,
         batch_size: int = 256,
         update_interval: int = 256,
         num_updates: int = 5,
         seed: int = 0,
         save_dir: Path = Path('runs')):

    print(update_interval)

    save_dir = save_dir / datetime.now().isoformat()

    create_path(save_dir)

    set_seed(seed)

    num_input = 384
    num_output = 1

    device = torch.device("cuda")
    agent = Agent(num_input, num_output, device, lr, momentum, replay_buffer_size)

    eps = 0.05
    batch_losses = []
    episode = 0
    it = 0

    board = chess.Board()
    game = chess.pgn.Game()
    moves = []

    expert = chess.engine.SimpleEngine.popen_uci("./stockfish")
    expert.analyse(board, chess.engine.Limit(time=0.001))['score'].relative
    white_reward = 0
    black_reward = 0
    og_len_episode = len_episode

    while episode < num_episodes:

        if episode < 5000:
            len_episode = 30
        elif episode < 10000:
            len_episode = 60
        else:
            len_episode - og_len_episode

        states = []
        rewards = []

        legal_moves = list(board.legal_moves)

        for m in legal_moves:
            color = board.turn
            board.push(m)
            states.append(
                torch.as_tensor(board_to_feature(board)).float()
            )
            reward = expert.analyse(board, chess.engine.Limit(time=0.001))['score'].pov(color)

            reward = reward.score(mate_score=3000) / 100

            rewards.append(reward - (white_reward if color else black_reward))
            board.pop()

        observations = torch.stack(states).float()

        idx = agent.action_step(observations, eps)
        action = legal_moves[idx]
        reward = rewards[idx]

        if color:
            white_reward = reward
        else:
            black_reward = reward

        board.push(action)
        moves.append(action)
        done = board.is_game_over() or it % len_episode == 0

        agent.replay_buffer.push(
            states[idx],
            torch.as_tensor(reward).float(),
            observations,
            int(done)
        )

        if it % update_interval == 0 and len(agent.replay_buffer) >= batch_size:

            errors = []
            for _ in range(num_updates):
                loss = agent.train_step(
                    batch_size,
                    gamma,
                    polyak
                )
                errors.append(loss.item())
                batch_losses.append(sum(errors) / len(errors))

            print(f'episode: {episode}, loss:{batch_losses[-1]:.4f}')

        it += 1

        if done:

            game.add_line(moves)
            with open(save_dir / f"game_ep{episode}.pgn", "w", encoding="utf-8") as f:
                exporter = chess.pgn.FileExporter(f)
                game.accept(exporter)

            board = chess.Board()
            game = chess.pgn.Game()
            moves = []
            episode += 1

    expert.quit()


if __name__ == '__main__':
    fire.Fire(main)
