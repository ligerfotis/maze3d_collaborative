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

    # create the experiment
    experiment = Experiment(maze, config=test_config)

    # set the goal
    goal = test_config["game"]["goal"]

    # training loop
    loop = test_config['Experiment']['loop']

    # Test loop
    experiment.test_human(goal)

    pg.quit()


if __name__ == '__main__':
    main(sys.argv[1:])
    exit(0)
