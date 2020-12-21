import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pip._vendor.distlib._backport import shutil


def plot_learning_curve(x, scores, figure_file):
    # running_avg = np.zeros(len(scores))
    # for i in range(len(running_avg)):
    #     running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, scores)
    plt.title('Total Rewards per Episode')
    plt.savefig(figure_file)


def plot_actions(x, actions, figure_file):
    plt.figure()
    plt.plot(x, actions)
    plt.title('Actions')
    plt.savefig(figure_file)


def plot(data, figure_file, x=None, title=None):
    plt.figure()
    if x is None:
        x = [i + 1 for i in range(len(data))]
    plt.plot(x, data)
    if title:
        plt.title(title)
    plt.savefig(figure_file)


def get_plot_and_chkpt_dir(load_checkpoint, load_checkpoint_name, discrete=False):
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    plot_dir = None
    if not load_checkpoint:
        if discrete:
            chkpt_dir = 'tmp/sac_discrete_' + timestamp
            plot_dir = 'plots/sac_discrete' + timestamp
        else:
            chkpt_dir = 'tmp/sac_' + timestamp
            plot_dir = 'plots/sac_' + timestamp
        # if not os.path.exists('tmp/sac'):
        #     os.makedirs('tmp/sac')
        #     chkpt_dir = 'tmp/sac'
        # else:
        #     os.makedirs(chkpt_dir)
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        shutil.copy('config.yaml', chkpt_dir)
    else:
        print("Loading Model from checkpoint {}".format(load_checkpoint_name))
        chkpt_dir = 'tmp/' + load_checkpoint_name

    return chkpt_dir, plot_dir, timestamp


def get_config(config_file='config.yaml'):
    try:
        with open(config_file) as file:
            yaml_data = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file')

    return yaml_data




def reward_function(env, observation, timedout):
    # For every timestep -1
    # For positioning between the flags with both legs touching fround and engines closed +100
    # Crush -100
    # Timed out -50
    # (Not Implemented) +5 for every leg touching (-5 for untouching)
    # (Not Implemented) +20 both touching
    if env.game_over or abs(observation[0]) >= 1.0:
        return -100, True

    leg1_touching, leg2_touching = [observation[6], observation[7]]
    # check if lander in flags and touching the ground
    if env.helipad_x1 < env.lander.position.x < env.helipad_x2 \
            and leg1_touching and leg2_touching:
        # solved
        return 200, True

    # if not done and timedout
    if timedout:
        return -50, True

    # return -1 for each time step
    return -1, False
