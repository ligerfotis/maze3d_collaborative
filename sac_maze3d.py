import csv
import time
from datetime import timedelta
import torch
from Maze3DEnv import Maze3D
from assets import *
from rl_models.sac_agent import Agent
from rl_models.utils import get_config, get_plot_and_chkpt_dir, reward_function, plot_learning_curve, plot


def main():
    # get configuration
    config = get_config()
    # creating environment
    maze = Maze3D()

    random_seed = None
    if random_seed:
        torch.manual_seed(random_seed)

    chkpt_dir, plot_dir, timestamp = get_plot_and_chkpt_dir(config['game']['load_checkpoint'],
                                                            config['game']['checkpoint_name'])
    sac = Agent(config=config, env=maze, input_dims=maze.observation_shape, n_actions=maze.action_space.shape,
                chkpt_dir=chkpt_dir)
    best_score = -100 - 1 * config['Experiment']['max_timesteps']
    best_score_episode = -1
    best_score_length = -1
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 1
    total_steps = 0

    training_epochs_per_update = 128
    action_history = []
    score_history = []
    episode_duration_list = []
    length_list = []
    grad_updates_durations = []
    info = {}
    if config['game']['load_checkpoint']:
        sac.load_models()
        # env.render(mode='human')

    max_episodes = config['Experiment']['max_episodes']
    max_timesteps = config['Experiment']['max_timesteps']
    start_experiment = time.time()
    # training loop
    for i_episode in range(1, max_episodes + 1):
        observation = maze.reset()
        timedout = False
        episode_reward = 0
        start = time.time()
        grad_updates_duration = 0
        for timestep in range(max_timesteps+1):
            total_steps += 1

            # if total_steps < start_training_step:  # Pure exploration
            #     action = random.randint(0, action_dim - 1)
            # else:  # Explore with actions_prob
            #     action = sac.choose_action(observation)

            # action = sac.choose_action(observation)
            # action = maze.action_space.sample()
            action = maze.action_space.sample()
            action_history.append(action)

            """
            Add the human part here
            """
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    maze.running = False
                # if event.type == pg.KEYDOWN:
                #     if event.key in maze.keys:
                #         action += maze.keys[event.key]
                # if event.type == pg.KEYUP:
                #     if event.key in maze.keys:
                #         action -= maze.keys[event.key]
            if timestep == max_timesteps:
                timedout = True

            observation_, reward, done = maze.step(action[0], timedout)

            sac.remember(observation, action, reward, observation_, done)
            if not config['game']['test_model']:
                sac.learn([observation, action, reward, observation_, done])
            observation = observation_

            # while 1:
            #     for event in pg.event.get():
            #         if event.type == pg.QUIT:
            #             maze.running = False
            #         # if event.type == pg.KEYDOWN:
            #         #     if event.key in maze.keys:
            #         #         action += maze.keys[event.key]
            #         # if event.type == pg.KEYUP:
            #         #     if event.key in maze.keys:
            #         #         action -= maze.keys[event.key]
            #     action = maze.action_space.sample()
            #     observation_, done = maze.step(action)
            #     if done:
            #         break
            if not config['game']['test_model']:
                # off policy learning
                start_grad_updates = time.time()
                update_cycles = config['Experiment']['update_cycles']

                if total_steps >= config['Experiment'][
                    'start_training_step'] and total_steps % sac.update_interval == 0:
                    for e in range(update_cycles):
                        sac.learn()
                        # sac.soft_update_target()
                end_grad_updates = time.time()
                grad_updates_duration += end_grad_updates - start_grad_updates

            # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
            #     sac.soft_update_target()

            running_reward += reward
            episode_reward += reward
            # if render:
            #     env.render()
            if done:
                break

        end = time.time()
        episode_duration = end - start
        episode_duration_list.append(episode_duration)
        score_history.append(episode_reward)
        avg_grad_updates_duration = grad_updates_duration / timestep
        grad_updates_durations.append(avg_grad_updates_duration)

        avg_ep_duration = np.mean(episode_duration_list[-100:])
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            best_score_episode = i_episode
            best_score_length = timestep
            if not config['game']['test_model']:
                sac.save_models()
        # for e in range(training_epochs_per_update):
        #     sac.learn()
        #     sac.soft_update_target()
        length_list.append(timestep)
        avg_length += timestep

        # logging
        if not config['game']['test_model']:
            log_interval = config['Experiment']['log_interval']
            if i_episode % log_interval == 0:
                avg_length = int(avg_length / log_interval)
                running_reward = int((running_reward / log_interval))

                print('Episode {} \t avg length: {} \t Total reward(last {} episodes): {} \t Best Score: {} \t avg '
                      'episode duration: {} avg grad updates duration: {}'.format(i_episode, avg_length, log_interval,
                                                                                  running_reward, best_score,
                                                                                  timedelta(seconds=avg_ep_duration),
                                                                                  timedelta(
                                                                                      seconds=avg_grad_updates_duration)))
            running_reward = 0
            avg_length = 0
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment)
    info['experiment_duration'] = experiment_duration
    info['best_score'] = best_score
    info['best_score_episode'] = best_score_episode
    info['best_score_length'] = best_score_length
    info['total_steps'] = total_steps

    print('Total Experiment time: {}'.format(experiment_duration))

    if not config['game']['test_model']:
        x = [i + 1 for i in range(len(score_history))]
        np.savetxt('tmp/sac_' + timestamp + '/scores.csv', np.asarray(score_history), delimiter=',')

        actions = np.asarray(action_history)
        # action_main = actions[0].flatten()
        # action_side = actions[1].flatten()
        x_actions = [i + 1 for i in range(len(actions))]
        # Save logs in files
        np.savetxt('tmp/sac_' + timestamp + '/actions.csv', actions, delimiter=',')
        # np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
        np.savetxt('tmp/sac_' + timestamp + '/epidode_durations.csv', np.asarray(episode_duration_list), delimiter=',')
        np.savetxt('tmp/sac_' + timestamp + '/avg_length_list.csv', np.asarray(length_list), delimiter=',')
        w = csv.writer(open('tmp/sac_' + timestamp + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])
        np.savetxt('tmp/sac_' + timestamp + '/grad_updates_durations.csv', grad_updates_durations, delimiter=',')

        plot_learning_curve(x, score_history, plot_dir + "/scores.png")
        # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
        # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
        plot(length_list, plot_dir + "/length_list.png", x=[i + 1 for i in range(max_episodes)])
        plot(episode_duration_list, plot_dir + "/epidode_durations.png", x=[i + 1 for i in range(max_episodes)])
        plot(grad_updates_durations, plot_dir + "/grad_updates_durations.png", x=[i + 1 for i in range(max_episodes)])

    pg.quit()


if __name__ == '__main__':
    main()
    exit(0)
