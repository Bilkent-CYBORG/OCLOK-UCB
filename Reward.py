"""
This class represents a worker reward (i.e. performance as stated in the paper)
"""


class Reward:
    def __init__(self, worker, performance):
        self.performance = performance
        self.context = worker.context
        self.worker = worker
