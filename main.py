import os

# disable cuda because joblib does not work well with cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # disable tensorflow printing logging messages

import multiprocessing
import pickle
import argparse
from gpflow.kernels import SquaredExponential
from scipy.interpolate import make_interp_spline

from problem_models.movielens_problem_model import MovielensProblemModel
from problem_models.synth_problem_model import SyntheticProblemModel

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from ACC_UCB import ACCUCB
from CC_MAB import CCMAB
from Hypercube import Hypercube
from benchmark_algo import Benchmark
from problem_models.fs_problem_model import FoursquareProblemModel

from oclok_ucb import OCLOK_UCB

sns.set_theme(style='whitegrid')

# taken from https://jwalton.info/Embed-Publication-Matplotlib-Latex
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.serif": 'Times New Roman',
    # # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 16,
    "font.size": 16,
    # # Make the legend/label fonts a little smaller
    # "legend.fontsize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
}
plt.rcParams.update(tex_fonts)

parser = argparse.ArgumentParser()
parser.add_argument("sim_type", type=str, help="Which simulation to run. 'sim_1' runs Simulation I (movie recommendation), "
                                               "'sim_2' runs Simulation II (Foursquare), "
                                               "and 'sim_3' runs Simulation II (varying "
                                               "arm dependency)")
parser.add_argument("--use_saved_dataset", default=False, action="store_true",
                    help="Whether to use the pre-generated"
                         "datasets on which the paper was "
                         "run to run the simulations. If "
                         "this is set to True, "
                         "then the pre-generated datasets "
                         "must be downloaded from the link "
                         "in the README.md file and put "
                         "into the root directory where "
                         "this script is.")

parser.add_argument("--only_plot", default=False, action="store_true",
                    help="If set to True, will NOT rerun simulations and only plot results from already run "
                         "simulations. If True, simulations must have been already run before at some point.")

parser.add_argument("--num_repeats", type=int, default=8, required=False,
                    help="Number of times to repeat the simulation and average over results.")
parser.add_argument("--num_threads", type=int, default=8, required=False,
                    help="Number of parallel processes to launch to run each independent run. Ideally should be a "
                         "divisor of num_repeats. If set to -1, as many processes as thread count will be launched.")

args = parser.parse_args()

# run types
MULTIPLE_ROUNDS = 'multiple_round'
SINGLE_ROUND = 'single_round'
MULTIPLE_KERNELS = 'multiple_kernels'

# problem model types
FOURSQUARE_MODEL = "fs"  # used in Simulations II
MOVIELENS_MODEL = "ml"  # used in Simulation I (dynamic probabilistic maximum coverage)

if args.sim_type == "sim_1":
    running_mode = SINGLE_ROUND
    model_type = MOVIELENS_MODEL
elif args.sim_type == "sim_2":
    running_mode = SINGLE_ROUND
    model_type = FOURSQUARE_MODEL
elif args.sim_type == "sim_3":
    running_mode = MULTIPLE_KERNELS
else:
    raise RuntimeError("sim_type must be one of 'sim_1', 'sim_2', or 'sim_3'")

use_generated_workers_in_paper = args.use_saved_dataset

num_threads_to_use = args.num_threads
if num_threads_to_use == -1:
    num_threads_to_use = int(multiprocessing.cpu_count())
use_saved_data = args.only_plot  # when True, the script simply plots the data of the most recently ran simulation, if available
# this means that no simulations are run when True.

num_times_to_run = args.num_repeats
if running_mode == MULTIPLE_KERNELS:
    num_rounds_arr = np.linspace(10, 300, 30).astype(int)
else:
    if model_type == FOURSQUARE_MODEL:
        num_rounds_arr = np.linspace(10, 250, 20).astype(int)
    else:
        num_rounds_arr = np.linspace(10, 400, 14).astype(int)

# foursquare params
num_std_to_show = 5
exp_num_workers = 100
max_num_workers = 150  # set this to a number s.t. Pr(num workers > max_num_workers) is very small ~1e-7.
noise_std = 0.1
fs_budget = 5

# movielens params
exp_left_nodes = 75  # i.e., expected number of movies in each round
exp_right_nodes = 200  # i.e., expected number of users in each round
movielens_budget = 3

delta = 0.05
context_dim = 3

# acc-ucb params
v1 = np.sqrt(context_dim)
v2 = 1
rho = 0.5
N = 2 ** context_dim
root_hypercube = Hypercube(1, np.full(context_dim, 0.5))  # this is called x_{0,1} in the paper

round_budget = fs_budget if model_type == FOURSQUARE_MODEL else movielens_budget

reference_algo = "Benchmark"
mc_name = "AOM-MC"
acc_name = "ACC-UCB"
mab_name = "CC-MAB"
gp_name = "O'CLOK-UCB"
bench_name = "Benchmark"

# for multi-kernel simulations seen in supplemental
kernel_lengthscales = [0.01, 0.05, 0.1, 0.5, 1]
kernel_list = [SquaredExponential(1, x) for x in kernel_lengthscales]


def run_one_try(problem_model, run_num, run_gp=True):
    oclock_kernel = gpflow.kernels.SquaredExponential(1, 1)
    if model_type == MOVIELENS_MODEL:
        inducing_pts = [1, 2, 4]
    else:
        inducing_pts = [20, 50, 100]

    if model_type == MOVIELENS_MODEL:
        problem_model.tim_graph_name = f"run_num_{run_num}"

    algo_result_dict = {}
    if run_gp:
        print('Running Benchmark...')
        bench_algo = Benchmark(problem_model, round_budget)
        algo_result_dict[bench_name] = bench_algo.run_algorithm()
        if model_type == MOVIELENS_MODEL:
            # save benchmark choices to be used later when computing regret
            problem_model.set_benchmark_superarm_list(algo_result_dict[bench_name]["bench_slate_list"])

        print(rf"Running {gp_name}...")
        ccgp_ucb_algo = OCLOK_UCB(problem_model, context_dim, round_budget, delta, max_num_workers,
                                  kernel=oclock_kernel, noise_variance=noise_std ** 2)
        algo_result_dict[rf"{gp_name}"] = ccgp_ucb_algo.run_algorithm()
        for inducing_pt in inducing_pts:
            print(
                f"Running S{gp_name}...")
            ccgp_ucb_algo = OCLOK_UCB(problem_model, context_dim, round_budget, delta, max_num_workers,
                                      kernel=oclock_kernel, use_sparse=True, num_inducing=inducing_pt,
                                      noise_variance=noise_std ** 2)
            algo_result_dict[
                rf"S{gp_name} ({inducing_pt} inducing pts.)"] = ccgp_ucb_algo.run_algorithm()

    print("Running ACC-UCB...")
    acc_ucb_algo = ACCUCB(problem_model, v1, v2, N, rho, root_hypercube, round_budget)
    algo_result_dict[acc_name] = acc_ucb_algo.run_algorithm()

    if model_type == FOURSQUARE_MODEL:
        cc_mab_algo = CCMAB(problem_model, root_hypercube.get_dimension(), round_budget)
        print('Running CC-MAB...')
        algo_result_dict[mab_name] = cc_mab_algo.run_algorithm()

    return algo_result_dict


def run_once_num_round(num_rounds):
    if model_type == FOURSQUARE_MODEL:
        problem_model = FoursquareProblemModel(num_rounds, exp_num_workers, use_generated_workers_in_paper,
                                               round_budget, noise_std)
    elif model_type == MOVIELENS_MODEL:
        problem_model = MovielensProblemModel(num_rounds, exp_left_nodes, exp_right_nodes,
                                              use_generated_workers_in_paper,
                                              movielens_budget)
    else:
        raise RuntimeError("No such model type!")

    # problem_model = SyntheticProblemModel(num_rounds, exp_num_workers, use_generated_workers_in_paper,
    #                                       round_budget, noise_std, context_dim,
    #                                       gpflow.kernels.SquaredExponential())
    # problem_model = GPProblemModel(num_rounds, max(exp_num_workers), root_hypercube.get_dimension(), use_generated_workers_in_paper)
    # problem_model = GowallaProblemModel(num_rounds, max(exp_num_workers), use_generated_workers_in_paper)
    # problem_model = TestProblemModel(num_rounds, max(exp_num_workers), use_generated_workers_in_paper)

    print("Running GP on {thread_count} threads".format(thread_count=num_threads_to_use))
    parallel_results = Parallel(n_jobs=num_threads_to_use)(
        delayed(run_one_try)(problem_model, i) for i in range(num_times_to_run))

    with open("{}_parallel_results_rounds_{}".format(model_type, num_rounds), 'wb') as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    return parallel_results


def run_for_diff_num_rounds():
    if not use_generated_workers_in_paper:  # load problem model with max num rounds
        if model_type == FOURSQUARE_MODEL:
            problem_model = FoursquareProblemModel(max(num_rounds_arr), exp_num_workers, False, round_budget, noise_std)
        elif model_type == MOVIELENS_MODEL:
            problem_model = MovielensProblemModel(max(num_rounds_arr), exp_left_nodes, exp_right_nodes, False,
                                                  movielens_budget)
        else:
            raise RuntimeError("No such model type!")
    parallel_results_list = []
    for num_rounds in tqdm(num_rounds_arr):
        problem_model = FoursquareProblemModel(num_rounds, exp_num_workers, True, round_budget, noise_std)
        if model_type == FOURSQUARE_MODEL:
            problem_model = FoursquareProblemModel(num_rounds, exp_num_workers, True, round_budget, noise_std)
        elif model_type == MOVIELENS_MODEL:
            problem_model = MovielensProblemModel(num_rounds, exp_left_nodes, exp_right_nodes, True, movielens_budget)
        else:
            raise RuntimeError("No such model type!")
        # problem_model = GPProblemModel(num_rounds, max(exp_num_workers), root_hypercube.get_dimension(), use_generated_workers_in_paper)
        # problem_model = GowallaProblemModel(num_rounds, max(exp_num_workers), use_generated_workers_in_paper)
        # problem_model = TestProblemModel(num_rounds, max(exp_num_workers), use_generated_workers_in_paper)

        print("Running GP on {thread_count} threads".format(thread_count=num_threads_to_use))
        print("Doing {} many rounds...".format(num_rounds))
        parallel_results = Parallel(n_jobs=num_threads_to_use)(
            delayed(run_one_try)(problem_model, i) for i in range(num_times_to_run))
        parallel_results_list.append(parallel_results)

        with open("{}_parallel_results_rounds_{}".format(model_type, num_rounds), 'wb') as output:
            pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    return parallel_results_list


def plot_cum_regret(results_list):
    algo_names = list(results_list[0][0].keys())
    num_Ts = len(results_list)
    cum_regret_arr = np.zeros((len(algo_names), len(results_list[0]), num_Ts))  # algo, repeat, T

    for i, results in enumerate(results_list):
        for j, result in enumerate(results):
            for k, algo_name in enumerate(algo_names):
                algo_dict = result[algo_name]
                final_regret = np.cumsum(algo_dict['regret_arr'])[-1]
                cum_regret_arr[k, j, i] = final_regret

    cum_regret_avg = cum_regret_arr.mean(axis=1)
    cum_regret_std = cum_regret_arr.std(axis=1)

    plt.figure(figsize=(6.4, 4))
    for i, algo_name in enumerate(algo_names):
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = cum_regret_avg[i], cum_regret_std[i]
            plt.plot(num_rounds_arr, mean, label=algo_name, color=color)
            plt.fill_between(num_rounds_arr, mean - std, mean + std, alpha=0.3, color=color)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.legend()
    plt.xlabel("Number of rounds ($T$)")
    plt.ylabel("Cumulative regret")
    plt.tight_layout()
    plt.savefig("cum_regret.pdf", bbox_inches='tight', pad_inches=0.03)


def get_reward_reg_time_from_result(parallel_results, algo_names):
    algo_reward_dict = {}
    algo_regret_dict = {}
    algo_time_dict = {}
    num_times_to_run = len(parallel_results)
    for i, entry in enumerate(parallel_results):
        for algo_name in algo_names:
            result = entry[algo_name]
            if algo_name not in algo_reward_dict:
                num_rounds = len(result['total_reward_arr'])
                algo_reward_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
                algo_regret_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
                algo_time_dict[algo_name] = np.zeros((num_times_to_run, num_rounds))
            algo_reward_dict[algo_name][i] = pd.Series(result['total_reward_arr']).expanding().mean().values
            # algo_reward_dict[algo_name][i] = np.cumsum(result['total_reward_arr'])
            algo_regret_dict[algo_name][i] = np.cumsum(result['regret_arr'])
            if algo_name != reference_algo:
                algo_time_dict[algo_name][i] = np.cumsum(result['time_taken_arr'])
        for algo_name in algo_names:
            if algo_name != reference_algo:
                algo_reward_dict[algo_name][i] /= algo_reward_dict[reference_algo][i]
    return algo_reward_dict, algo_regret_dict, algo_time_dict


def plot_reward_and_time(parallel_results):
    algo_names = list(parallel_results[0].keys())
    num_rounds = len(parallel_results[0][algo_names[0]]['total_reward_arr'])

    plot_names = algo_names

    algo_reward_dict, algo_regret_dict, algo_time_dict = get_reward_reg_time_from_result(parallel_results, algo_names)

    algo_reward_avg_dict = {}
    algo_reward_std_dict = {}
    algo_regret_avg_dict = {}
    algo_regret_std_dict = {}
    algo_time_avg_dict = {}
    algo_time_std_dict = {}
    for algo_name in algo_names:
        algo_reward_avg_dict[algo_name] = algo_reward_dict[algo_name].mean(axis=0)
        algo_reward_std_dict[algo_name] = 1 * algo_reward_dict[algo_name].std(axis=0)
        algo_regret_avg_dict[algo_name] = algo_regret_dict[algo_name].mean(axis=0)
        algo_regret_std_dict[algo_name] = 1 * algo_regret_dict[algo_name].std(axis=0)
        algo_time_avg_dict[algo_name] = algo_time_dict[algo_name].mean(axis=0)
        algo_time_std_dict[algo_name] = 1 * algo_time_dict[algo_name].std(axis=0)

        xnew = np.arange(1, num_rounds + 1)

        # smooth
        # xnew = np.linspace(1, num_rounds, 75)
        spl = make_interp_spline(range(1, num_rounds + 1), algo_reward_avg_dict[algo_name], k=3)
        algo_reward_avg_dict[algo_name] = spl(xnew)

        spl = make_interp_spline(range(1, num_rounds + 1), algo_reward_std_dict[algo_name], k=3)
        algo_reward_std_dict[algo_name] = spl(xnew)

        algo_reward_avg_dict[algo_name][0] = algo_reward_std_dict[algo_name][0] = 0

    # PLOT AVERAGE REWARD
    plt.figure(2, figsize=(6.4, 3.5))
    for i, algo_name in enumerate(algo_names):
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = algo_reward_avg_dict[algo_name], algo_reward_std_dict[algo_name]
            plt.plot(xnew, mean, label=plot_names[i], color=color)  # TODO REMOVE
            plt.fill_between(xnew, mean - std, mean + std, alpha=0.3, color=color)

    plt.legend()
    # plt.xlim(0, 200)
    plt.ylim(0, 1)  # We need to do this b/c otherwise the legend was not seen
    plt.xlabel("Arriving task $(t)$")
    plt.ylabel("Average task reward divided by\nbenchmark reward up to task $t$")
    plt.tight_layout()
    plt.savefig("avg_reward.pdf", bbox_inches='tight', pad_inches=0.02)

    # PLOT TIME TAKEN
    plt.figure(3, figsize=(6.4, 4))
    for algo_name in algo_names:
        if algo_name != reference_algo:
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            mean, std = algo_time_avg_dict[algo_name], algo_time_std_dict[algo_name]
            plt.plot(range(1, 1 + num_rounds), mean, label=algo_name.replace("CCGP-UCB", "O'CLOK-UCB"), color=color)
            plt.fill_between(range(1, 1 + num_rounds), mean - std, mean + std, alpha=0.3, color=color)

    plt.legend()
    # plt.xlim(0, 200)
    # plt.ylim(0.95, 1)  # We need to do this b/c otherwise the legend was not seen
    plt.xlabel("Arriving task $(t)$")
    plt.ylabel("Time taken (s)")
    plt.tight_layout()
    plt.savefig("time_taken.pdf", bbox_inches='tight', pad_inches=0.02)


def plot_multiple_kernel_reward(final_round_results):
    algo_names = list(final_round_results[0][0].keys())
    num_rounds = len(final_round_results[0][0][algo_names[0]]['total_reward_arr'])
    num_kernels = len(kernel_list)

    rewards_arr_avg = np.zeros((len(algo_names), num_kernels, num_rounds))
    rewards_arr_std = np.zeros((len(algo_names), num_kernels, num_rounds))

    for i, results in enumerate(final_round_results):
        algo_reward_dict, algo_regret_dict, algo_time_dict = get_reward_reg_time_from_result(results, algo_names)
        for j, algo_name in enumerate(algo_names):
            rewards_arr_avg[j, i, :] = algo_reward_dict[algo_name].mean(axis=0)
            rewards_arr_std[j, i, :] = algo_reward_dict[algo_name].std(axis=0)

    f, axes = plt.subplots(2, 3, figsize=(15, 9))
    x = np.arange(1, num_rounds + 1)
    for i, algo_name in enumerate([x for x in algo_names if x != reference_algo]):
        index = np.unravel_index(i, (2, 3))
        algo_index = algo_names.index(algo_name)
        for j in range(num_kernels):
            color = next(axes[index]._get_lines.prop_cycler)['color']
            mean, std = rewards_arr_avg[algo_index, j], rewards_arr_std[algo_index, j]
            axes[index].plot(x, mean,
                             label="Outcome kernel $l= {:.2f}$".format(kernel_lengthscales[j]),
                             linewidth=2, color=color)
            axes[index].fill_between(x, mean - std, mean + std, alpha=0.3, linewidth=2, color=color)

        axes[index].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[index].set_title(algo_name, fontsize=16)
        axes[index].legend()
        axes[index].set_ylim([-0.3, 1.0])
        axes[index].set_xlabel("Round number $(t)$", fontsize=16)
        # if i == 0:
        axes[index].set_ylabel("Average task reward divided\nby benchmark reward up to $t$", fontsize=16)
    # axes[-1, -1].set_visible(False)
    f.tight_layout()
    f.savefig("multi_kernel_reward.pdf", bbox_inches='tight', pad_inches=0.02)

    # for i, algo_name in enumerate(algo_names):
    #     if algo_name != reference_algo:
    #         plt.figure()
    #         for j in range(num_kernels):
    #             plt.errorbar(range(1, num_rounds + 1), rewards_arr_avg[i, j], yerr=rewards_arr_std[i, j],
    #                          label="Outcome kernel $l= {:.2f}$".format(kernel_lengthscales[j]), capsize=2, linewidth=2)
    #         plt.legend()
    #         plt.xlabel("Round number $(t)$")
    #         plt.ylabel("Average task reward divided\nby benchmark reward up to $t$")
    #         plt.ylim([-0.3, 1.0])
    #         plt.tight_layout()
    #         plt.savefig("{}_reward.pdf".format(algo_name), bbox_inches='tight', pad_inches=0.02)
    # plt.show()


def get_reg_from_multi_round(results_list, gp_alg_names, algo_names, num_Ts):
    non_gp_alg_names = [x for x in algo_names if x not in gp_alg_names]
    cum_regret_arr = np.zeros((len(algo_names), len(results_list[0]), num_Ts))  # algo, repeat, T

    for i, results in enumerate(results_list):
        for j, result in enumerate(results):
            if i == len(results_list) - 1:  # last result is one with most num rounds so GP algs will be included
                for k, algo_name in enumerate(gp_alg_names):
                    algo_dict = result[algo_name]
                    for m, final_T in enumerate(num_rounds_arr):
                        cum_regret_arr[algo_names.index(algo_name), j, m] = np.cumsum(algo_dict['regret_arr'])[
                            final_T - 1]
            for k, algo_name in enumerate(non_gp_alg_names):
                algo_dict = result[algo_name]
                final_regret = np.cumsum(algo_dict['regret_arr'])[-1]
                cum_regret_arr[algo_names.index(algo_name), j, i] = final_regret

    return cum_regret_arr


def plot_multiple_kernel_reg(all_results_list):  # kernel -> different Ts -> different repeats -> algo result
    algo_names = [x for x in all_results_list[0][-1][0].keys() if x != reference_algo]

    gp_alg_names = [x for x in algo_names if gp_name in x]
    num_kernels = len(kernel_list)
    num_Ts = len(all_results_list[0])

    regret_arr_avg = np.zeros((len(algo_names), num_kernels, num_Ts))
    regret_arr_std = np.zeros((len(algo_names), num_kernels, num_Ts))

    for i, results in enumerate(all_results_list):
        cum_regret_arr = get_reg_from_multi_round(results, gp_alg_names, algo_names, num_Ts)
        regret_arr_avg[:, i, :] = cum_regret_arr.mean(axis=1)
        regret_arr_std[:, i, :] = cum_regret_arr.std(axis=1)

    f, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, algo_name in enumerate(algo_names):
        index = np.unravel_index(i, (2, 3))
        for j in range(num_kernels):
            color = next(axes[index]._get_lines.prop_cycler)['color']
            mean, std = regret_arr_avg[i, j], regret_arr_std[i, j]
            axes[index].plot(num_rounds_arr, mean,
                             label="Outcome kernel $l= {:.2f}$".format(kernel_lengthscales[j]),
                             linewidth=2, color=color)
            axes[index].fill_between(num_rounds_arr, mean - std, mean + std, alpha=0.3, linewidth=2, color=color)

        axes[index].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[index].set_title(algo_name, fontsize=16)
        axes[index].legend()
        axes[index].set_ylim([-10, 3.2e3])
        t = axes[index].yaxis.get_offset_text()
        t.set_x(-0.05)
        axes[index].set_xlabel("Number of rounds $(T)$", fontsize=16)
        axes[index].set_ylabel("Cumulative regret", fontsize=16)
    f.tight_layout()
    plt.savefig("multi_kernel_reg.pdf", bbox_inches='tight', pad_inches=0.02)


def run_with_diff_kernel(rounds_arr, kernel_list):
    all_results_list = []
    for i, kernel in enumerate(tqdm(kernel_list)):
        if not use_generated_workers_in_paper:  # load problem model with max num rounds
            problem_model = SyntheticProblemModel(max(rounds_arr), exp_num_workers, False,
                                                  round_budget, noise_std, context_dim,
                                                  kernel, "synthetic_df_{}".format(i))
        parallel_results_list = []
        for num_rounds in rounds_arr:
            problem_model = SyntheticProblemModel(num_rounds, exp_num_workers, True,
                                                  round_budget, noise_std, context_dim,
                                                  kernel, "synthetic_df_{}".format(i))

            print("Doing {} many rounds...".format(num_rounds))

            # only run GP algos with the largest number of rounds because GP algos are not affected by choice of num_rounds
            parallel_results = Parallel(n_jobs=num_threads_to_use)(
                delayed(run_one_try)(problem_model, run_num=0, run_gp=num_rounds == max(rounds_arr)) for _ in
                range(num_times_to_run))
            parallel_results_list.append(parallel_results)
        all_results_list.append(parallel_results_list)

        with open('{}_multiple_kernels_{}'.format(model_type, i), 'wb') as output:
            pickle.dump(parallel_results_list, output, pickle.HIGHEST_PROTOCOL)
        all_results_list.append(parallel_results_list)
    return all_results_list


if __name__ == '__main__':
    if not use_saved_data:
        if running_mode == MULTIPLE_ROUNDS:
            parallel_results_list = run_for_diff_num_rounds()
            plot_cum_regret(parallel_results_list)
            plot_reward_and_time(parallel_results_list[-1])
        elif running_mode == SINGLE_ROUND:
            parallel_results = run_once_num_round(max(num_rounds_arr))
            plot_reward_and_time(parallel_results)
        elif running_mode == MULTIPLE_KERNELS:
            all_results_list = run_with_diff_kernel(num_rounds_arr, kernel_list)
            plot_multiple_kernel_reg(all_results_list)
            plot_multiple_kernel_reward([x[-1] for x in all_results_list])

    else:
        if running_mode == MULTIPLE_ROUNDS:
            parallel_results_list = []
            for num_rounds in num_rounds_arr:
                with open('{}_parallel_results_rounds_{}'.format(model_type, num_rounds), 'rb') as input_file:
                    parallel_results = pickle.load(input_file)
                parallel_results_list.append(parallel_results)
            plot_cum_regret(parallel_results_list)
            plot_reward_and_time(parallel_results_list[-1])

        elif running_mode == SINGLE_ROUND:
            with open('{}_parallel_results_rounds_{}'.format(model_type, max(num_rounds_arr)), 'rb') as input_file:
                parallel_results = pickle.load(input_file)
            plot_reward_and_time(parallel_results)
        elif running_mode == MULTIPLE_KERNELS:
            parallel_results_list = []
            for i in range(len(kernel_list)):
                with open('{}_multiple_kernels_{}'.format(model_type, i), 'rb') as input_file:
                    parallel_results = pickle.load(input_file)
                parallel_results_list.append(parallel_results)
            plot_multiple_kernel_reg(parallel_results_list)
            plot_multiple_kernel_reward([x[-1] for x in parallel_results_list])
    plt.show()
