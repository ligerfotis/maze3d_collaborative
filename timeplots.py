import numpy as np
from math import sqrt
from statistics import stdev, mean
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def get_episode_cumulative_time_list(file_name):
    episodes_durations = np.genfromtxt(file_name, delimiter=',')
    acc_sum = 0
    accumulative_episode_duration_list = []
    for i in range(0, len(episodes_durations), 10):
        acc_sum += sum(episodes_durations[i:i + 10])
        accumulative_episode_duration_list.append(acc_sum)

    return accumulative_episode_duration_list


def get_grad_time_list(file_name):
    grad_durations = np.genfromtxt(file_name, delimiter=',')
    acc_sum = 0
    accumulative_episode_duration_list = []
    for i in range(len(grad_durations)):
        acc_sum += grad_durations[i]
        accumulative_episode_duration_list.append(acc_sum)

    return accumulative_episode_duration_list

colors = ["g", "b", "r", "m", "navy", "darkorange"]
colors_light = ["limegreen", "cornflowerblue", "lightcoral", "violet", "slateblue", "wheat"]


def cummulative_time_plot(filename_list, legend_names, figure_file=None):
    fig, ax = plt.subplots()
    axes = plt.gca()
    # axes.set_ylim([start, end])
    # colors = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        for c, file_name_sublist in enumerate(filename_list):
            epidode_durations_merge_lists, grad_updates_durations_merge_lists = [], []
            for file_name in file_name_sublist:
                epidode_durations = get_episode_cumulative_time_list(file_name+"epidode_durations.csv")
                grad_updates_durations = get_grad_time_list(file_name+"grad_updates_durations.csv")

                epidode_durations_merge_lists.append(epidode_durations)
                grad_updates_durations_merge_lists.append(grad_updates_durations)

            epidode_durations_means, grad_updates_means, x_axis = [0], [0], []
            total_time_means, stds = [], []
            for i in range(len(epidode_durations_merge_lists[0])):
                epidode_durations_data_points, grad_updates_data_points = [], []
                for j in range(len(epidode_durations_merge_lists)):
                    epidode_durations_data_points.append(epidode_durations_merge_lists[j][i])
                    grad_updates_data_points.append(grad_updates_durations_merge_lists[j][i])
                epidode_durations_means.append(mean(epidode_durations_data_points)/60)
                grad_updates_means.append(mean(grad_updates_data_points) / 60)
                total_time_means.append(mean(epidode_durations_data_points)/60+mean(grad_updates_data_points)/60)
                stds.append(stdev(epidode_durations_data_points)/60+stdev(grad_updates_data_points)/60 / sqrt(len(epidode_durations_means+grad_updates_means)))
                x_axis.append(i+1)
            # x_axis.append(7)
            epidode_durations_means, grad_updates_means, x_axis = np.asarray(epidode_durations_means), np.asarray(grad_updates_means), np.asarray(x_axis)
            total_time_means, stds = np.asarray(total_time_means), np.asarray(stds)
            # ax.plot(x_axis, epidode_durations_means, c=colors[c], label=legend_names[c], marker='*')
            # ax.plot(x_axis, grad_updates_means, c=colors[c], label=legend_names[c], marker='*')
            ax.plot(x_axis, total_time_means, c=colors[c], label=legend_names[c], marker='*')
            ax.fill_between(x_axis, total_time_means - stds, total_time_means + stds, facecolor=colors_light[c], alpha=0.5)

        # ax.yaxis.set_ticks(np.arange(start, end, 10))
        ax.set_ylabel('Cumulative Total Time Elapsed(min)')
        ax.set_xlabel('Offline Gradient Update Sessions')
        plt.grid()
        plt.legend(loc='upper left')
        plt.savefig(figure_file)


dir = "figures/times/"

file_name_1 = "tmp/expert_alg1_online_154K_every10_sparse2_1/"
file_name_2 = "tmp/expert_alg1_online_154K_every10_sparse2_2/"
file_name_3 = "tmp/expert_alg1_online_154K_every10_sparse2_3/"

file_name_4 = "tmp/expert_alg1_offline_154K_every10_sparse2_1/"
file_name_5 = "tmp/expert_alg1_offline_154K_every10_sparse2_2/"
file_name_6 = "tmp/expert_alg1_offline_154K_every10_sparse2_3/"

file_name_7 = 'tmp/expert_alg1_online_28K_every10_sparse2_1/'
file_name_8 = 'tmp/expert_alg1_online_28K_every10_sparse2_2/'
file_name_9 = 'tmp/expert_alg1_online_28K_every10_sparse2_3/'

file_name_10 = 'tmp/expert_alg1_offline_28K_every10_sparse2_1/'
file_name_11 = 'tmp/expert_alg1_offline_28K_every10_sparse2_2/'
file_name_12 = 'tmp/expert_alg1_offline_28K_every10_sparse2_3/'

file_name_13 = 'tmp/expert_alg1_online_28K_every10_sparse2_descending_1/'
file_name_14 = 'tmp/expert_alg1_online_28K_every10_sparse2_descending_2/'
file_name_15 = 'tmp/expert_alg1_online_28K_every10_sparse2_descending_3/'

file_name_16 = 'tmp/expert_alg1_offline_28K_every10_sparse2_descending_1/'
file_name_17 = 'tmp/expert_alg1_offline_28K_every10_sparse2_descending_2/'
file_name_18 = 'tmp/expert_alg1_offline_28K_every10_sparse2_descending_3/'

file_name_1t = 'tmp/thanasis/loop1_O_O_a_154K_every10_Sparse_2_normal_thanasis_1/'
file_name_2t = 'tmp/thanasis/loop1_O_O_a_154K_every10_Sparse_2_normal_thanasis_2/'
file_name_3t = 'tmp/thanasis/loop1_O_O_a_154K_every10_Sparse_2_normal_thanasis_3/'

file_name_4t = 'tmp/thanasis/loop1_O_a_154K_every10_Sparse_2_normal_thanasis_1/'
file_name_5t = 'tmp/thanasis/loop1_O_a_154K_every10_Sparse_2_normal_thanasis_2/'
file_name_6t = 'tmp/thanasis/loop1_O_a_154K_every10_Sparse_2_normal_thanasis_3/'

file_name_7t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_normal_thanasis_1/'
file_name_8t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_normal_thanasis_2/'
file_name_9t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_normal_thanasis_3/'

file_name_10t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_normal_thanasis_1/'
file_name_11t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_normal_thanasis_2/'
file_name_12t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_normal_thanasis_3/'

file_name_13t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_descending_thanasis_1/'
file_name_14t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_descending_thanasis_2/'
file_name_15t = 'tmp/thanasis/loop1_O_O_a_28K_every10_Sparse_2_descending_thanasis_3/'

file_name_16t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_descending_thanasis_1/'
file_name_17t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_descending_thanasis_2/'
file_name_18t = 'tmp/thanasis/loop1_O_a_28K_every10_Sparse_2_descending_thanasis_3/'

episodes_duration_file = "epidode_durations.csv"
grad_updates_file = "grad_updates_durations.csv"

# grad_durations = np.genfromtxt(main_dir + grad_updates_file, delimiter=',')

legend_names = ["O-O-a 154K", "O-a 154K", "O-O-a 28K", "O-a 28K", "O-O-a 28K Descending", "O-a 28K Descending"]

cummulative_time_plot([[file_name_1, file_name_2, file_name_3, file_name_1t, file_name_2t, file_name_3t],
                       [file_name_4, file_name_5, file_name_6, file_name_4t, file_name_5t, file_name_6t],
                       [file_name_7, file_name_8, file_name_9, file_name_7t, file_name_8t, file_name_9t],
                       [file_name_10, file_name_11, file_name_12, file_name_10t, file_name_11t, file_name_12t],
                       [file_name_13, file_name_14, file_name_15, file_name_13t, file_name_14t, file_name_15t],
                       [file_name_16, file_name_17, file_name_18, file_name_16t, file_name_17t, file_name_18t]],
                      legend_names,
                      figure_file=dir + "Cumulative Time 156K vs 28K vs 28K Descending 2 users")




















