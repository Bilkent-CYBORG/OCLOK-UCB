import time

import math

from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from math import sqrt

import gpflow
import numpy as np
from tqdm import tqdm
from typing import List

from problem_models import ProblemModel

from Hypercube import Hypercube
from UcbNode import UcbNode


# def find_nodes_in_bound_list(boundaries_arr, leaves):
#     """This function takes the midpoint context of each leaf, subtracts it from all intervals and then multiplies the
#     interval start and end points of each dimension together. If the result is negative, that means that dimension
#     contains the context."""
#     node_list = []
#     # boundary_list: (dim, interval, start/end)
#     for leaf in leaves:
#         result = (boundaries_arr.T - leaf.get_mid_context()).T
#         result = result[:, :, 0] * result[:, :, 1]
#         if np.count_nonzero(result < 0) == boundaries_arr.shape[0]:
#             node_list.append(leaf)
#     return node_list

def is_cube_in_bound_list(hypercube, boundaries_arr):
    h_starting = hypercube.center - hypercube.length / 2
    h_ending = hypercube.center + hypercube.length / 2

    starting_result = (boundaries_arr.T - h_starting).T
    ending_result = (h_ending - boundaries_arr.T).T
    result = starting_result[:, :, 1] * ending_result[:, :, 0]
    return np.all(np.any(result > 0, axis=1))


# def find_nodes_in_bound_list(boundaries_arr, root_node):
#     """This function takes the midpoint context of each leaf, subtracts it from all intervals and then multiplies the
#     interval start and end points of each dimension together. If the result is negative, that means that dimension
#     contains the context."""
#     node_list = []
#     # boundary_list: (dim, interval, start/end)
#     for leaf in leaves:
#         result = (boundaries_arr.T - leaf.get_mid_context()).T
#         result = result[:, :, 0] * result[:, :, 1]
#         if np.count_nonzero(result < 0) == boundaries_arr.shape[0]:
#             node_list.append(leaf)
#     return node_list

def find_nodes_in_bound_from_root(boundaries_arr, root_node, output):
    is_leaf = root_node.children is None
    if is_leaf:
        output.append(root_node)
    else:
        for child in root_node.children:
            intersection_list = [is_cube_in_bound_list(h, boundaries_arr) for h in child.hypercube_list]
            if np.any(intersection_list):
                find_nodes_in_bound_from_root(boundaries_arr, child, output)


def find_nodes_in_bound_from_leaves(boundaries_arr, leaves) -> List[UcbNode]:
    node_list = []
    for leaf in leaves:
        intersection_list = [is_cube_in_bound_list(h, boundaries_arr) for h in leaf.hypercube_list]
        if np.any(intersection_list):
            node_list.append(leaf)
    return node_list


"""
This class represents the ACCGP-UCB algorithm that is presented in the paper.
"""


def sample_points_from_node(node_to_play, num_samples):
    """
    Generates AT LEAST num_samples many samples from the region of the given node
    """
    ns_per_cube = int(np.ceil(num_samples / len(node_to_play.hypercube_list)))
    samples_arr = np.zeros((ns_per_cube * len(node_to_play.hypercube_list), node_to_play.dimension))
    for i, hypercube in enumerate(node_to_play.hypercube_list):
        low = hypercube.center - hypercube.length / 2
        high = hypercube.center + hypercube.length / 2
        samples_arr[ns_per_cube * i:(i + 1) * ns_per_cube] = np.random.uniform(low, high, size=(
            ns_per_cube, node_to_play.dimension))
    return samples_arr


def check_context_in_b_list(context, available_boundaries):
    for boundaries_arr in available_boundaries:
        result = (boundaries_arr.T - context).T
        result = result[:, :, 0] * result[:, :, 1]
        if np.count_nonzero(result < 0) == boundaries_arr.shape[0]:
            return True
    return False


class OCLOK_UCB:
    problem_model: ProblemModel

    def __init__(self, problem_model: ProblemModel, dim, budget, delta, max_arriving_arms,
                 batch_mode=False, use_marginal_rewards=False,
                 use_sparse=False, num_inducing=100, optimize=False, mean_function=None,
                 kernel=gpflow.kernels.Matern52(),
                 noise_variance=1):
        self.max_arriving_arms = max_arriving_arms
        self.optimize = optimize
        self.use_marginal_rewards = use_marginal_rewards
        self.batch_mode = batch_mode
        self.problem_model = problem_model
        self.num_inducing = num_inducing
        X = np.zeros((1, dim))
        Y = np.zeros((1, 1))
        Z = X[:, :].copy()
        if use_sparse:
            self.gp_model = gpflow.models.SGPR(data=(X, Y), kernel=kernel, mean_function=mean_function,
                                               noise_variance=noise_variance, inducing_variable=Z)
        else:
            self.gp_model = gpflow.models.GPR(data=(X, Y), kernel=kernel, mean_function=mean_function,
                                              noise_variance=noise_variance)
        self.max_iter = 10
        self.budget = budget
        self.dim = dim
        self.num_rounds = problem_model.num_rounds
        self.delta = delta
        self.use_sparse = use_sparse

    def set_model_data(self, model, train_X, train_Y, dim):
        train_X_np = np.array(train_X).reshape((-1, dim))
        train_Y_np = np.array(train_Y).reshape((-1, 1))
        model.data = data_input_to_tensor((train_X_np, train_Y_np))
        if self.use_sparse:
            num_inducing = min(train_X_np.shape[0], self.num_inducing)
            inducing_indices = np.random.choice(range(train_X_np.shape[0]), num_inducing, replace=False)
            Z = train_X_np[inducing_indices, :].copy()
            model.inducing_variable = inducingpoint_wrapper(Z)

    def beta(self, t):
        m = self.max_arriving_arms
        return 2 * np.log(m * (t ** 2) * (np.pi ** 2) / (3 * self.delta))

    def run_algorithm(self):
        self.num_rounds = self.problem_model.num_rounds
        total_reward_arr = np.zeros(self.num_rounds)
        regret_arr = np.zeros(self.num_rounds)
        opt_logs = []
        time_taken_arr = np.zeros(self.num_rounds)
        train_X = []
        train_Y = []

        for t in tqdm(range(1, self.num_rounds + 1)):
            starting_time = time.time()
            available_arms = self.problem_model.get_available_arms(t)

            if self.batch_mode:
                sqrt_beta_t = np.sqrt(self.beta(t))
                new_train_X = train_X.copy()
                new_train_Y = train_Y.copy()

                # first compute all arms posterior mean
                posterior_means = np.array(
                    [self.gp_model.predict_f(arm.context.reshape(1, -1))[0] for arm in available_arms]).reshape(-1)

                slate_ids = []  # ids of arms corresponding to available arms
                # pick arms so to maximize marginal reward
                for i in range(self.budget):
                    self.set_model_data(self.gp_model, new_train_X, new_train_Y, self.dim)
                    posterior_std = np.array(
                        [self.gp_model.predict_f(arm.context.reshape(1, -1))[1] for arm in available_arms]).reshape(-1)
                    index_list = posterior_means + sqrt_beta_t * np.sqrt(posterior_std)
                    if not self.use_marginal_rewards:
                        index_list[slate_ids] = -1
                    if len(slate_ids) == 0 or not self.use_marginal_rewards:
                        # pick first arm by looking at highest index
                        best_arm_idx = index_list.argmax()
                    else:
                        marginal_rewards = np.full(len(index_list), -1.)
                        slate_indices = index_list[slate_ids]
                        new_slate_indices = np.zeros(len(slate_indices) + 1)
                        new_slate_indices[0:len(slate_indices)] = slate_indices
                        for i, arm_index in enumerate(index_list):
                            if i not in slate_ids:
                                new_slate_indices[-1] = arm_index
                                marginal_rewards[i] = self.problem_model.reward_fun(new_slate_indices) - \
                                                      self.problem_model.reward_fun(slate_indices)
                        best_arm_idx = marginal_rewards.argmax()
                    slate_ids.append(best_arm_idx)
                    new_train_X.append(available_arms[best_arm_idx].context)
                    new_train_Y.append(0.)
                slate = [available_arms[idx] for idx in slate_ids]

                # reset gp model
                self.set_model_data(self.gp_model, train_X, train_Y, self.dim)
            else:
                # rank based on index
                rand_arm_indices = np.arange(len(available_arms))  # removed randomness cause TIM
                # rand_arm_indices = np.random.choice(len(available_arms), size=len(available_arms), replace=False)
                # index_list = np.zeros(len(available_arms))
                # for i, arm_id in enumerate(rand_arm_indices):
                #     arm = available_arms[arm_id]
                #     index_list[i] = self.get_arm_index(arm, t)
                index_list = np.array([self.get_arm_index(arm, t) for arm in available_arms])

                idx_to_play = self.problem_model.oracle(self.budget, index_list, t)
                slate = [available_arms[idx] for idx in rand_arm_indices[idx_to_play]]

            rewards = self.problem_model.play_arms(t, slate)  # Returns a list of Reward objects

            # Store reward obtained
            total_reward_arr[t - 1] = self.problem_model.get_total_reward(rewards, t)
            regret_arr[t - 1] = self.problem_model.get_regret(t, self.budget, slate)

            # Update the GP model
            train_X.extend([arm.context for arm in slate])
            train_Y.extend([reward.performance for reward in rewards])

            self.set_model_data(self.gp_model, train_X, train_Y, self.dim)

            # opt_log = self.optimizer.minimize(self.gp_model.training_loss, self.gp_model.trainable_variables,
            #                                   options=dict(maxiter=self.max_iter))
            # if self.optimize:
            #     for _ in range(self.max_iter):
            #         self.optimizer.minimize(self.gp_model.training_loss, self.gp_model.trainable_variables)
            # opt_logs.append(opt_log)
            time_taken_arr[t - 1] = time.time() - starting_time

        # with open('CCGP-UCB-model', 'wb') as output:
        #     pickle.dump(self.gp_model, output, pickle.HIGHEST_PROTOCOL)
        return {
            'time_taken_arr': time_taken_arr,
            'total_reward_arr': total_reward_arr,
            'regret_arr': regret_arr,
            'gp_model': gpflow.utilities.freeze(self.gp_model)
        }

    def get_arm_index(self, arm, t):
        beta = self.beta(t)
        mean, var = self.gp_model.predict_f(arm.context.reshape(1, -1))
        return mean.numpy().item() + np.sqrt(beta * var.numpy().item())

    def calc_confidence(self, num_times_node_played):
        if num_times_node_played == 0:
            return float('inf')
        return sqrt(2 * math.log(self.num_rounds) / num_times_node_played)


if __name__ == '__main__':
    # TODO DELETE
    boundary = np.array([[[0.1, 0.7], [0, 0]], [[0.6, 0.8], [0.1, 0.2]]])
    hypercube1 = Hypercube(0.5, np.array([0.25, 0.25]))
    is_cube_in_bound_list(hypercube1, boundary)
