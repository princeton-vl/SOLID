import time
import torch

from tabulate import tabulate

tabulate.PRESERVE_WHITESPACE = True

ENABLED = True

class TimerNode(object):
    def __init__(self, nodes=None):
        self.timer = Timer()
        self.nodes = nodes if nodes is not None else {}

    def __getitem__(self, key):
        if key not in self.nodes:
            self.nodes[key] = TimerNode()
        return self.nodes[key]

    def __enter__(self):
        self.timer.tic()

    def __exit__(self, exc_type, exc_value, tb):
        self.timer.toc()


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        if not ENABLED:
            return

        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()

    def toc(self, average=True):
        if not ENABLED:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_value, tb):
        self.toc()


def prepare_table(timers, space, prefix):
    results = [
        [
            "-" * space + " " + prefix,
            timers.timer.average_time,
            timers.timer.total_time,
        ]
    ]
    if len(timers.nodes) == 0:
        return results

    for key, value in timers.nodes.items():
        results += prepare_table(
            value,
            space + 2,
            key,
        )
    return results


def print_timers(timers):
    table = prepare_table(timers, 0, "all")
    print(tabulate(
        table,
        headers=["Section", "Avg Time", "Tot Time"],
        tablefmt="pretty",
        colalign=("left", "right", "right"),
        floatfmt=".2f",
    ))
