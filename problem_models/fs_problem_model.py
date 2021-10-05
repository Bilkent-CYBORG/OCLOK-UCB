# import gmaps
import pickle
import random
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

# IMPORTANT: if you want to generate the Foursquare dataset from scratch, you MUST have the Wolfram Engine installed and also download the exported LearnedDistribution object (called tkyLd.wmlf) and set WMLF_LEARNED_DIST_PATH to the absolute path of the tkyLd.wmlf file. Lastly, you must install the Wolfram Python client and import it.
# WMLF_LEARNED_DIST_PATH = /home/user/Downloads/tkyLd.wmlf
# from wolframclient.evaluation import WolframLanguageSession


saved_df_name = "fs_tky_simulation_df"  # file where the simulation-ready dataframe will be saved
DISTANCE_THRESHOLD = np.sqrt(0.5)



def extract_random_task_and_workers(df):
    num_tasks = df.index[-1]
    ran_task = np.random.randint(1, num_tasks + 1)
    rows = df.loc[ran_task]
    loc_mean_list = np.concatenate(
        (np.stack(rows['context'].values)[:, 3:5], rows['true_mean'].to_numpy().reshape(-1, 1)), axis=1).tolist()


def save_task_loc(df):
    """
    Saves the task locations as a list to be used for map plotting. Note that task location is scaled to [0,1].
    """
    df = df.loc[~df.index.duplicated(keep='first')]
    location_list = np.stack(df['context'].values)[:, 0:2].tolist()

    with open(city_name + '_task_loc', 'wb') as output:
        pickle.dump(location_list, output, pickle.HIGHEST_PROTOCOL)


def context_to_mean_fun(context):
    """
    context[0] = worker-task distance
    context[1] = task loc
    context[2] = task context
    """
    return norm.pdf(context[0], loc=0, scale=0.4) * np.sqrt(context[1] * context[2]) / norm.pdf(0, loc=0, scale=0.4)



def filter_workers(worker_id_loc_pair, task_loc, distance_thresh):
    """
    Filters the workers based on the distance condition given in the paper
    Parameters
    ----------
    worker_id_loc_pair
    task_loc
    distance_thresh

    Returns
    -------

    """
    return list(
        filter(lambda id_loc_pair: np.linalg.norm(id_loc_pair[1] - task_loc) < distance_thresh, worker_id_loc_pair))


class FoursquareProblemModel(ProblemModel):
    df: pd.DataFrame

    def __init__(self, num_rounds, exp_avai_workers, use_saved, budget, noise_std, is_uniform=True, use_gmm=True):
        super().__init__(num_rounds)
        self.noise_std = noise_std
        self.budget = budget
        uni_str = "_uni" if is_uniform else "_nuni"
        if not use_saved:
            uni_df, nuni_df = self.initialize_df(exp_avai_workers, use_gmm)
            uni_df.set_index('time', inplace=True)
            nuni_df.set_index('time', inplace=True)
            with open(saved_df_name + "_uni", 'wb') as output:
                pickle.dump(uni_df, output, pickle.HIGHEST_PROTOCOL)
            with open(saved_df_name + "_nuni", 'wb') as output:
                pickle.dump(nuni_df, output, pickle.HIGHEST_PROTOCOL)

            self.df = uni_df if is_uniform else nuni_df
        else:
            with open(saved_df_name + uni_str, 'rb') as input_file:
                self.df = pickle.load(input_file)

        self.num_workers = len(self.df.loc[1:self.num_rounds].index)
        extract_random_task_and_workers(self.df)

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
        return np.log(1 + bench_reward_sum) - np.log(1 + algo_reward_sum)

    def get_total_reward(self, rewards, t=None):
        reward_sum = 0
        for reward in rewards:
            reward_sum += reward.performance
        if reward_sum <= -1:
            reward_sum = 0
        return np.log(1 + reward_sum)

    def reward_fun(self, outcome_arr, t=None):
        reward_sum = np.sum(outcome_arr)
        if reward_sum <= -1:
            reward_sum = 0
        return np.log(1 + reward_sum)

    def play_arms(self, t, slate):
        reward_list = []
        df = self.df.loc[t]
        for worker in slate:
            # first check if available
            # performance = np.random.binomial(1, df.iloc[worker.unique_id]['true_mean']) * available
            performance = df.iloc[worker.unique_id]['true_mean'] + np.random.normal(0, self.noise_std)
            reward_list.append(Reward(worker, performance))
        return reward_list

    def oracle(self, K, g_list, t=None):
        return np.argsort(g_list)[-K:]

    def get_task_budget(self, t):
        return self.df.loc[t].iloc[0]['task_budget']

    @staticmethod
    def non_uniform_rnd(prob_num_dict: dict):
        key_arr = np.array(list(prob_num_dict.keys()))[:, 1]
        key_cumsum = key_arr.cumsum()
        temp = np.random.rand()
        for i, key in enumerate(key_arr):
            if i == 0 and 0 <= temp < key:
                return prob_num_dict[(i, key)]
            elif key_cumsum[i - 1] <= temp < key_cumsum[i]:
                return prob_num_dict[(i, key)]

    def initialize_df(self, exp_avai_workers, use_gmm):
        print("Generating workers...")
        if use_gmm:
            with open('gmm_{}'.format(city_name), 'rb') as input_file:
                gmm = pickle.load(input_file)
            session = WolframLanguageSession()
            session.evaluate(f'ld=Import["{WMLF_LEARNED_DIST_PATH}"];')
        else:
            with open(fs_loader.saved_df_filename, 'rb') as input_file:
                df = pickle.load(input_file)
                available_index_set = set(range(len(df)))
        uni_row_list = []
        nuni_row_list = []

        for time in tqdm(range(1, self.num_rounds + 1)):
            # non uniform task context
            uni_task_context = np.random.uniform()
            nuni_task_context = self.non_uniform_rnd({(0, 0.60): np.random.uniform(0, 0.6),
                                                      (1, 0.20): np.random.uniform(0.6, 0.95),
                                                      (2, 0.20): np.random.uniform(0.95, 1.0)})

            # sample num available workers from Po distribution
            num_avai_workers = 0
            while num_avai_workers <= self.budget:
                num_avai_workers = np.random.poisson(exp_avai_workers)

            if use_gmm:
                task_location = np.random.uniform(0, 1, 2)
                # worker_locs = gmm.sample(5 * num_avai_workers)[0]
                worker_locs = session.evaluate(f'RandomVariate[ld,{5*num_avai_workers}]')
                avail_worker_locs = list(
                    filter(lambda loc: np.linalg.norm(loc - task_location) < DISTANCE_THRESHOLD, worker_locs))

                for worker_location in avail_worker_locs[0:num_avai_workers]:
                    worker_location = np.clip(worker_location, 0, 1)
                    uni_worker_battery = np.random.uniform()
                    nuni_worker_battery = self.non_uniform_rnd({(0, 0.60): np.random.uniform(0, 0.6),
                                                                (1, 0.20): np.random.uniform(0.6, 0.95),
                                                                (2, 0.20): np.random.uniform(0.95, 1.0)})

                    # compact context is a list of each context, whereas context is the concatenation of each context
                    # context is what will be used by ACC-UCB and CC-MAB to perform the discretization
                    distance = np.linalg.norm(task_location - worker_location) / DISTANCE_THRESHOLD

                    uni_compact_context = [distance, uni_task_context, uni_worker_battery]
                    uni_true_mean = context_to_mean_fun(uni_compact_context)

                    nuni_compact_context = [distance, nuni_task_context, nuni_worker_battery]
                    nuni_true_mean = context_to_mean_fun(nuni_compact_context)

                    context_uni = np.hstack(uni_compact_context)
                    context_nuni = np.hstack(nuni_compact_context)

                    uni_row_list.append((time, context_uni, uni_true_mean))
                    nuni_row_list.append((time, context_nuni, nuni_true_mean))
            else:
                # randomly pick users from dataset
                avail_id_loc_pairs = []
                while len(avail_id_loc_pairs) < num_avai_workers:
                    task_location = np.random.uniform(0, 1, 2)
                    picked_indeces = random.sample(available_index_set, num_avai_workers)
                    avail_id_loc_pairs = list(filter(lambda idx: np.linalg.norm(
                        np.array([df.iloc[idx]['lat'], df.iloc[idx]['long']]) - task_location) <
                                                                 DISTANCE_THRESHOLD, picked_indeces))
                for idx in avail_id_loc_pairs:
                    available_index_set.remove(idx)
                    uni_worker_battery = np.random.uniform()
                    nuni_worker_battery = self.non_uniform_rnd({(0, 0.60): np.random.uniform(0, 0.6),
                                                                (1, 0.20): np.random.uniform(0.6, 0.95),
                                                                (2, 0.20): np.random.uniform(0.95, 1.0)})
                    worker_location = np.array([df.iloc[idx]['lat'], df.iloc[idx]['long']])

                    # compact context is a list of each context, whereas context is the concatenation of each context
                    # context is what will be used by ACC-UCB and CC-MAB to perform the discretization
                    # distance = np.linalg.norm(task_location - worker_location) / DISTANCE_THRESHOLD

                    uni_compact_context = [worker_location, task_location, uni_task_context, uni_worker_battery]
                    uni_true_mean = context_to_mean_fun(uni_compact_context)

                    nuni_compact_context = [worker_location, task_location, nuni_task_context, nuni_worker_battery]
                    nuni_true_mean = context_to_mean_fun(nuni_compact_context)

                    context_uni = np.hstack(uni_compact_context)
                    context_nuni = np.hstack(nuni_compact_context)

                    uni_row_list.append((time, context_uni, uni_true_mean))
                    nuni_row_list.append((time, context_nuni, nuni_true_mean))

        session.terminate()
        return pd.DataFrame(uni_row_list,
                            columns=['time', 'context', 'true_mean']), \
               pd.DataFrame(nuni_row_list,
                            columns=['time', 'context', 'true_mean'])

