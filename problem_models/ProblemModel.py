from abc import ABC

"""
This abstract class represents a problem model that the ACC-UCB algorithm will run on.
"""


class ProblemModel(ABC):
    def __init__(self, num_rounds):
        self.num_workers = None  # must be set in a subclass
        self.num_rounds = num_rounds

    def get_available_arms(self, t):
        pass

    def oracle(self, K, g_list, t=None):
        pass

    def play_arms(self, t, slate):
        pass

    def reward_fun(self, outcome_arr, t=None):
        pass

    def get_regret(self, t, budget, slate):
        pass

    def get_task_budget(self, t):
        pass

    def get_total_reward(self, rewards, t=None):
        pass
