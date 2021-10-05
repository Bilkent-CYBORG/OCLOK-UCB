import matplotlib.pyplot as plt
import pickle
import random

import gpflow
import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

import fs_loader
from fs_loader import city_name
from ProblemModel import ProblemModel
from Reward import Reward
from Arm import Arm

"""
This file contains code for the Foursuqre problem model. The FoursuqreProblemModel class had functions to 
provide available arms, play the arms, calculate regret, and etc.
"""
MAX_NUM_SAMPLES = 6000


class SyntheticProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_avai_workers, use_saved, budget, noise_std, context_dim, kernel, saved_df_name):
        super().__init__(num_rounds)
        self.kernel = gpflow.utilities.freeze(kernel)
        self.context_dim = context_dim
        self.noise_std = noise_std
        self.rng = np.random.default_rng(341)
        self.weight_mat = self.rng.random(context_dim)

        self.budget = budget
        if not use_saved:
            num_samples = min(MAX_NUM_SAMPLES, exp_avai_workers * num_rounds)
            self.disc_contexts = np.random.random((num_samples, 3))
            self.disc_samples = self.context_to_mean_fun(self.disc_contexts)
            df = self.initialize_df(exp_avai_workers)
            df.set_index('time', inplace=True)
            with open(saved_df_name, 'wb') as output:
                pickle.dump(df, output, pickle.HIGHEST_PROTOCOL)

            self.df = df
        else:
            with open(saved_df_name, 'rb') as input_file:
                self.df = pickle.load(input_file)

        self.num_workers = len(self.df.loc[1:self.num_rounds].index)

    def context_to_mean_fun(self, context):
        context_mat = context.reshape(-1, self.context_dim)
        var = self.kernel(context_mat)
        mean = np.zeros(context_mat.shape[0])
        samples = np.random.multivariate_normal(mean, var, 1).T
        return samples

    def get_available_arms(self, t):
        # Construct a list of Arm objects
        arm_list = []
        for _, row in self.df.loc[t].iterrows():
            arm_list.append(Arm(len(arm_list), row['context'], row['true_mean']))
        return arm_list

    # log log reward
    # def get_regret(self, t, budget, slate):
    #     df = self.df.loc[t]
    #     highest_means = df['true_mean'].nlargest(budget)  # greedy oracle
    #     algo_reward_sum = bench_reward_sum = 0
    #     for i, worker in enumerate(slate):
    #         bench_reward_sum += np.log(1 + AVAILABILITY_PROB * highest_means.iloc[i])
    #         algo_reward_sum += np.log(1 + AVAILABILITY_PROB * df.iloc[worker.unique_id]['true_mean'])
    #     return np.log(1 + bench_reward_sum) - np.log(1 + algo_reward_sum)
    #
    # def get_total_reward(self, rewards):
    #     reward_sum = 0
    #     for reward in rewards:
    #         reward_sum += np.log(1 + reward.performance)
    #     return np.log(1 + reward_sum)

    # log reward
    def get_regret(self, t, budget, slate):
        df = self.df.loc[t]
        highest_means = df['true_mean'].nlargest(budget)  # greedy oracle
        algo_reward_sum = bench_reward_sum = 0
        for i, worker in enumerate(slate):
            bench_reward_sum += highest_means.iloc[i]
            algo_reward_sum += df.iloc[worker.unique_id]['true_mean']
        return bench_reward_sum - algo_reward_sum

    def get_total_reward(self, rewards, t=None):
        reward_sum = 0
        for reward in rewards:
            reward_sum += reward.performance
        return reward_sum

    def reward_fun(self, outcome_arr, t=None):
        reward_sum = np.sum(outcome_arr)
        return reward_sum

    # def get_regret(self, t, budget, slate):
    #     df = self.df.loc[t]
    #     highest_means = df['true_mean'].nlargest(budget)
    #     algo_mean_prod = 1
    #     bench_mean_prod = 1
    #     for arm in slate:
    #         algo_mean_prod *= 1 - df.iloc[arm.unique_id]['true_mean']
    #     for mean in highest_means:
    #         bench_mean_prod *= 1 - mean
    #
    #     return algo_mean_prod - bench_mean_prod
    #
    # def get_total_reward(self, rewards):
    #     reward_sum = 0
    #     for reward in rewards:
    #         reward_sum += reward.performance  # Total reward is lin sum
    #     if reward_sum >= 1:
    #         return 1
    #     return 0

    # linear reward
    # def get_regret(self, t, budget, slate):
    #     df = self.df.loc[t]
    #     highest_means = df['true_mean'].nlargest(budget)  # greedy oracle
    #     algo_reward_sum = bench_reward_sum = 0
    #     for i, worker in enumerate(slate):
    #         bench_reward_sum += AVAILABILITY_PROB * highest_means.iloc[i]
    #         algo_reward_sum += AVAILABILITY_PROB * df.iloc[worker.unique_id]['true_mean']
    #     return bench_reward_sum - algo_reward_sum
    #
    # def get_total_reward(self, rewards):
    #     reward_sum = 0
    #     for reward in rewards:
    #         reward_sum += reward.performance
    #     return reward_sum

    # def get_regret(self, t, budget, slate):
    #     df = self.df.loc[t]
    #     highest_means = df['true_mean'].nlargest(budget)
    #     algo_mean_prod = 1
    #     bench_mean_prod = 1
    #     for arm in slate:
    #         algo_mean_prod *= 1 - df.iloc[arm.unique_id]['true_mean'] * AVAILABILITY_PROB
    #     for mean in highest_means:
    #         bench_mean_prod *= 1 - mean * AVAILABILITY_PROB
    #
    #     return algo_mean_prod - bench_mean_prod
    #
    # def get_total_reward(self, rewards):
    #     reward_sum = 0
    #     for reward in rewards:
    #         reward_sum += reward.performance  # Total reward is lin sum
    #     if reward_sum >= 1:
    #         return 1
    #     return 0

    def play_arms(self, t, slate):
        reward_list = []
        df = self.df.loc[t]
        for worker in slate:
            performance = df.iloc[worker.unique_id]['true_mean'] + np.random.normal(0, self.noise_std)
            reward_list.append(Reward(worker, performance))
        return reward_list

    def oracle(self, K, g_list, t=None):
        return np.argsort(g_list)[-K:]

    def initialize_df(self, exp_avai_workers):
        print("Generating workers...")
        row_list = []
        for time in tqdm(range(1, self.num_rounds + 1)):
            num_avai_workers = 0
            while num_avai_workers <= self.budget:
                num_avai_workers = np.random.poisson(exp_avai_workers)
            for i in range(num_avai_workers):
                sampled_context_ind = np.random.randint(0, self.disc_contexts.shape[0])
                context = self.disc_contexts[sampled_context_ind]
                true_mean = self.disc_samples[sampled_context_ind].item()
                row_list.append((time, context, true_mean))

        return pd.DataFrame(row_list, columns=['time', 'context', 'true_mean'])


if __name__ == '__main__':
    # test = FoursquareProblemModel(3000, 100, False, True, True)

    # df['']
    print('donerooni')
