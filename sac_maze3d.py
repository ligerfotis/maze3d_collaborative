import time
from datetime import timedelta
from experiment import Experiment
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent
from rl_models.utils import get_config, get_plot_and_chkpt_dir
from maze3D.utils import save_logs_and_plot


def main():
    # get configuration
    config = get_config()
    # creating environment
    maze = Maze3D()

    discrete = config['SAC']['discrete']

    chkpt_dir, plot_dir, timestamp = get_plot_and_chkpt_dir(config)
    if discrete:
        if config['Experiment']['loop'] == 1:
            buffer_max_size = config['Experiment']['loop_1']['buffer_memory_size']
            update_interval = config['Experiment']['loop_1']['learn_every_n_episodes']
            scale = config['Experiment']['loop_1']['reward_scale']
        else:
            buffer_max_size = config['Experiment']['loop_2']['buffer_memory_size']
            update_interval = config['Experiment']['loop_2']['learn_every_n_timesteps']
            scale = config['Experiment']['loop_2']['reward_scale']

        sac = DiscreteSACAgent(config=config, env=maze, input_dims=maze.observation_shape,
                               n_actions=maze.action_space.actions_number,
                               chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                               reward_scale=scale)
    else:
        sac = Agent(config=config, env=maze, input_dims=maze.observation_shape, n_actions=maze.action_space.shape,
                    chkpt_dir=chkpt_dir)
    experiment = Experiment(config, maze, sac)
    start_experiment = time.time()

    # training loop
    loop = config['Experiment']['loop']
    if loop == 1:
        experiment.loop_1()
    else:
        experiment.loop_2()
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('Total Experiment time: {}'.format(experiment_duration))
    experiment.df.to_pickle(plot_dir + '/training_logs.pkl')
    if not config['game']['test_model']:
        if loop == 1:
            total_games = experiment.max_episodes
        else:
            total_games = experiment.game
        save_logs_and_plot(experiment, chkpt_dir, plot_dir, total_games)
        experiment.save_info(chkpt_dir, experiment_duration, total_games)
    pg.quit()


if __name__ == '__main__':
    main()
    exit(0)
