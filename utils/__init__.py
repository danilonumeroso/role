import random
import numpy as np
import torch
import datetime
import json
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(elapsed):
    # Format as hh:mm:ss:microseconds
    time = datetime.timedelta(seconds=elapsed)
    return f'{time}'


def create_path(path: Path):
    p = Path(path)

    if not p.exists():
        p.mkdir(parents=True)


def to_json(save_path, obj):
    f = open(save_path, "w")
    json.dump(obj, f, indent=2)


def from_json(path):
    f = open(path, "r")
    return json.load(f)
