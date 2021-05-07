import time
from utils import format_time


class Timer:

    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, t, v, traceback):
        elapsed = time.time() - self.start
        print(f'{self.name}: {format_time(elapsed)}')
        return True
