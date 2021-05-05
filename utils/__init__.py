import random
import numpy as np
import torch
import datetime

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(elapsed):
    # Format as hh:mm:ss:microseconds
    time = datetime.timedelta(seconds=elapsed)
    return f'{time}'
