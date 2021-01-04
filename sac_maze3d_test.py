import time
from datetime import timedelta
from experiment import Experiment
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from rl_models.sac_agent import Agent
from rl_models.sac_discrete_agent import DiscreteSACAgent
from rl_models.utils import get_config, get_plot_and_chkpt_dir, get_sac_agent, get_test_plot_and_chkpt_dir
from maze3D.utils import save_logs_and_plot
import sys


def main(argv):
    # get configuration
    test_config = get_config(argv[0])

    # creating environment
    maze = Maze3D(config_file=argv[0])

    # load the checkpoint and create test plot directories for this experiment
    test_plot_dir, load_checkpoint_name, config = get_test_plot_and_chkpt_dir(test_config)

    # create the SAC agent
    sac = get_sac_agent(config, maze, chkpt_dir=load_checkpoint_name)

    # create the experiment
    experiment = Experiment(config, maze, sac, load_models=True)
    start_experiment = time.time()

    # set the goal
    goal = config["game"]["goal"]

    # Test Loop
    experiment.test_loop()
    # # training loop
    # loop = config['Experiment']['loop']
    # if loop == 1:
    #     # Experiment 1
    #     experiment.loop_1(goal)
    # else:
    #     # Experiment 2
    #     experiment.loop_2(goal)

    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment - experiment.duration_pause_total)

    print('Total Experiment time: {}'.format(experiment_duration))

    if config["game"]["save"]:
        # save training logs to a pickle file
        experiment.df.to_pickle(plot_dir + '/training_logs.pkl')

        if not config['game']['test_model']:
            total_games = experiment.max_episodes if loop == 1 else experiment.game
            # save rest of the experiment logs and plot them
            save_logs_and_plot(experiment, chkpt_dir, plot_dir, total_games)
            experiment.save_info(chkpt_dir, experiment_duration, total_games, goal)
    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
