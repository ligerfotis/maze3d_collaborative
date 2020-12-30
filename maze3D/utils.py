# right_down = [104, 104]
# center = [-24, 8]
import math
import numpy as np

from rl_models.utils import plot_learning_curve, plot

center = [0, 0]
left_down = [-104, -104]
left_up = [-104, 73]
right_down = [73, -104]

#################
goal = left_down
################


def checkTerminal(ball):
    # print("Ball x:{} y{}".format(ball.x, ball.y))
    if goal == right_down:
        if ball.x > goal[0] and ball.y < goal[1]:
            return True
    elif goal == left_up:
        if ball.x < goal[0] and ball.y > goal[1]:
            return True
    elif goal == left_down:
        if ball.x < goal[0] and ball.y < goal[1]:
            return True
    elif goal == center:
        if ball.x < 0 and ball.y < 0:
            return True
    else:
        return False


def get_distance_from_goal(ball):
    return math.sqrt(math.pow(ball.x - goal[0], 2) + math.pow(ball.y - goal[1], 2))


def convert_actions(actions):
    action = []
    if actions[0] == 1:
        action.append(1)
    elif actions[1] == 1:
        action.append(2)
    else:
        action.append(0)
    if actions[2] == 1:
        action.append(1)
    elif actions[3] == 1:
        action.append(2)
    else:
        action.append(0)
    return action

def save_logs_and_plot(experiment, chkpt_dir, plot_dir, max_episodes):

        x = [i + 1 for i in range(len(experiment.score_history))]
        np.savetxt(chkpt_dir + '/scores.csv', np.asarray(experiment.score_history), delimiter=',')

        actions = np.asarray(experiment.action_history)
        # action_main = actions[0].flatten()
        # action_side = actions[1].flatten()
        x_actions = [i + 1 for i in range(len(actions))]
        # Save logs in files
        np.savetxt(chkpt_dir + '/actions.csv', actions, delimiter=',')
        # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
        np.savetxt(chkpt_dir + '/epidode_durations.csv', np.asarray(experiment.episode_duration_list), delimiter=',')
        np.savetxt(chkpt_dir + '/avg_length_list.csv', np.asarray(experiment.length_list), delimiter=',')

        np.savetxt(chkpt_dir + '/grad_updates_durations.csv', experiment.grad_updates_durations, delimiter=',')

        plot_learning_curve(x, experiment.score_history, plot_dir + "/scores.png")
        # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
        # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
        plot(experiment.length_list, plot_dir + "/length_list.png", x=[i + 1 for i in range(max_episodes)])
        plot(experiment.episode_duration_list, plot_dir + "/epidode_durations.png",
             x=[i + 1 for i in range(max_episodes)])
        plot(experiment.grad_updates_durations, plot_dir + "/grad_updates_durations.png",
             x=[i + 1 for i in range(len(experiment.grad_updates_durations))])

