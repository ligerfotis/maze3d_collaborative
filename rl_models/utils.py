import os
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
from pip._vendor.distlib._backport import shutil

from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent


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


def get_plot_and_chkpt_dir(config):

    load_checkpoint, load_checkpoint_name, discrete = [config['game']['load_checkpoint'],
                                                       config['game']['checkpoint_name'], config['SAC']['discrete']]
    loop = str(config['Experiment']['loop'])
    now = datetime.now()
    timestamp = str(now.strftime("%Y%m%d_%H-%M-%S"))
    plot_dir = None
    if not load_checkpoint:
        if 'chkpt_dir' in config["SAC"].keys():
            chkpt_dir = 'tmp/' + config['SAC']['chkpt_dir']
            plot_dir = 'plots/' + config['SAC']['chkpt_dir']
        else:
            if discrete:
                chkpt_dir = 'tmp/sac_discrete_loop' + loop + "_" + timestamp
                plot_dir = 'plots/sac_discrete_loop' + loop + "_" + timestamp
            else:
                chkpt_dir = 'tmp/sac_loop' + loop + "_" + timestamp
                plot_dir = 'plots/sac_loop' + loop + "_" + timestamp
            # if not os.path.exists('tmp/sac'):
            #     os.makedirs('tmp/sac')
            #     chkpt_dir = 'tmp/sac'
            # else:
            #     os.makedirs(chkpt_dir)
        if not os.path.exists(chkpt_dir):
            os.makedirs(chkpt_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        shutil.copy('config_sac.yaml', chkpt_dir)
    else:
        print("Loading Model from checkpoint {}".format(load_checkpoint_name))
        chkpt_dir = 'tmp/' + load_checkpoint_name

    return chkpt_dir, plot_dir, timestamp, load_checkpoint_name


def get_test_plot_and_chkpt_dir(test_config):
    # get the config from the train folder
    config = None

    load_checkpoint_name = test_config['checkpoint_name']

    print("Loading Model from checkpoint {}".format(load_checkpoint_name))
    participant = test_config['participant']
    test_plot_dir = 'test/sac_discrete_' + participant + "/"
    if not os.path.exists(test_plot_dir):
        os.makedirs(test_plot_dir)

    assert os.path.exists(load_checkpoint_name)

    return test_plot_dir, load_checkpoint_name, config


def get_config(config_file='config_sac.yaml'):
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


def get_sac_agent(config, env, chkpt_dir=None):
    discrete = config['SAC']['discrete']
    if discrete:
        if config['Experiment']['loop'] == 1:
            buffer_max_size = config['Experiment']['loop_1']['buffer_memory_size']
            update_interval = config['Experiment']['loop_1']['learn_every_n_episodes']
            scale = config['Experiment']['loop_1']['reward_scale']
        else:
            buffer_max_size = config['Experiment']['loop_2']['buffer_memory_size']
            update_interval = config['Experiment']['loop_2']['learn_every_n_timesteps']
            scale = config['Experiment']['loop_2']['reward_scale']

        if config['game']['agent_only']:
            # up: 1, down:2, left:3, right:4, upleft:5, upright:6, downleft: 7, downright:8
            action_dim = pow(2, env.action_space.actions_number)
        else:
            action_dim = env.action_space.actions_number
        sac = DiscreteSACAgent(config=config, env=env, input_dims=env.observation_shape,
                               n_actions=action_dim,
                               chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                               reward_scale=scale)
    else:
        sac = Agent(config=config, env=env, input_dims=env.observation_shape, n_actions=env.action_space.shape,
                    chkpt_dir=chkpt_dir)
    return sac
