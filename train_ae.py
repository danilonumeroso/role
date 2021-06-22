import torch
import chess
import fire

from random import random, uniform
from utils import create_path
from torch.nn import functional as F
from models.role.ae import AE
from models.random_player import RandomPlayer
from pathlib import Path
from utils.features import board_to_feat
from torch.utils.data import TensorDataset, DataLoader


def l1_loss(submodules, x):
    L = 0
    for m in submodules:
        x = F.relu(m(x))
        L += x.abs().mean()

    return L


def train(model, submodules, loader, optimizer, N, device):
    model.train()
    total_loss = 0

    for x, *_ in loader:
        x = x.to(device)
        y = model(x)

        loss = F.binary_cross_entropy(y, x) + l1_loss(submodules, x)
        total_loss += loss.item() * len(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / N


@torch.no_grad()
def eval(model, submodules, loader, N, device):
    model.eval()
    total_loss = 0
    for x, *_ in loader:
        x = x.to(device)
        y = model(x)

        loss = F.binary_cross_entropy(y, x) + l1_loss(submodules, x)
        total_loss += loss.item() * len(x)

    return total_loss / N


def main(batch_size: int = 128,
         num_kernels: int = 16,
         num_hidden: int = 1024,
         num_epochs: int = 100,
         lr: float = 1e-4,
         decay: float = 1e-4,
         z_dim: int = 4096,
         save_dir: Path = Path('./data')):

    data = torch.load(save_dir / 'board_positions.pth')
    num_channels = data.shape[1]

    # data = list(torch.unbind(data))
    # print(data[0].shape)
    # input()

    N = len(data)

    len_val = N // 10
    len_tr = N - len_val

    tr_data = data[:len_tr]
    vl_data = data[len_tr:]

    tr_set = TensorDataset(tr_data)
    vl_set = TensorDataset(vl_data)

    tr_loader = DataLoader(tr_set,
                           batch_size=batch_size,
                           shuffle=True)

    vl_loader = DataLoader(vl_set,
                           batch_size=batch_size,
                           shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE(in_dim=(8, 8),
               num_channels=num_channels,
               num_kernels=num_kernels,
               kernel_size=(3, 3),
               num_hidden=num_hidden,
               z_dim=z_dim,)

    submodules = list(model.children())

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    best = 100

    for epoch in range(num_epochs):

        tr_loss = train(model, submodules, tr_loader, optimizer, len_tr, device)
        vl_loss = eval(model, submodules, vl_loader, len_val, device)

        if vl_loss < best:
            print("BEST")
            best = vl_loss
            torch.save(model, save_dir / "ae.pth")

        print(f'Epoch {epoch:4d}> tr_loss: {tr_loss:.4f}, vl_loss: {vl_loss:.4f}')


def collect_data(data_len: int,
                 expert_path: Path = Path("./stockfish"),
                 save_dir: Path = Path('./data')):

    create_path(save_dir)
    stockfish = chess.engine.SimpleEngine.popen_uci(expert_path)
    random_player = RandomPlayer()
    data = []

    def go(board, rand_t):
        player = (stockfish if random() > rand_t else random_player)
        m = player.play(board, chess.engine.Limit(time=uniform(1e-6, 1e-4))).move

        board.push(m)
        return board_to_feat(board), board.occupied

    positions = []

    board = chess.Board()
    data.append(board_to_feat(board))
    positions.append(board.occupied)

    for i in range(data_len):

        if i < data_len // 25:
            rand_t = 0.25
        elif i < data_len // 60:
            rand_t = 0.5
        else:
            rand_t = 0.8

            feat, occupied = go(board, rand_t)

        if board.occupied in positions:
            continue

        positions.append(occupied)
        data.append(feat)

        if board.is_game_over():
            board = chess.Board()

    stockfish.quit()
    data = torch.stack(data)
    torch.save(data, save_dir / "board_positions.pth")


if __name__ == '__main__':
    fire.Fire(collect_data)
