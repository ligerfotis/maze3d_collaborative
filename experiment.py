import csv
import math
import random
from statistics import mean
import pandas as pd
import time
from datetime import timedelta
from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *
from maze3D.config import pause
import numpy as np
from tqdm import tqdm
from maze3D.utils import convert_actions
from maze3D.config import left_down, right_down, left_up, center
from pympler.tracker import SummaryTracker

column_names = ["actions_x", "actions_y", "tray_rot_x", "tray_rot_y", "tray_rot_vel_x", "tray_rot_vel_y",
                "ball_pos_x", "ball_pos_y", "ball_vel_x", "ball_vel_y"]
stdev = .1
tracker = SummaryTracker()

class Experiment:
    def __init__(self, environment, agent=None, load_models=False, config=None):
        self.train_fps_list = []
        self.test_fps_list = []
        self.test_step_duration_list = []
        self.online_update_duration_list = []
        self.step_duration_list = []
        self.counter = 0
        self.test = 0
        self.config = config
        self.env = environment
        self.agent = agent
        self.best_score = None
        self.best_reward = None
        self.best_score_episode = -1
        self.best_score_length = -1
        self.total_steps = 0
        self.action_history = []
        self.score_history = []
        self.episode_duration_list = []
        self.length_list = []
        self.grad_updates_durations = []
        self.test_length_list = []
        self.test_score_history = []
        self.test_episode_duration_list = []
        self.discrete = config['SAC']['discrete'] if 'SAC' in config.keys() else None
        self.second_human = config['game']['second_human'] if 'game' in config.keys() else None
        self.duration_pause_total = 0
        if load_models:
            self.agent.load_models()
        self.df = pd.DataFrame(columns=column_names)
        self.df_test = pd.DataFrame(columns=column_names)
        self.max_episodes = None
        self.max_timesteps = None
        self.avg_grad_updates_duration = 0
        self.human_actions = None
        self.agent_action = None
        self.total_timesteps = None
        self.max_timesteps_per_game = None
        self.save_models = True
        self.game = None
        self.test_max_timesteps = self.config['Experiment']['test_loop']['max_timesteps'] if 'test_loop' in config['Experiment'].keys() else None
        self.test_max_episodes = self.config['Experiment']['test_loop']['max_games'] if 'test_loop' in config['Experiment'].keys() else None
        self.update_cycles = None

        # Experiment 1 loop

    def loop_1(self, goal):
        # Experiment 1 loop
        flag = True
        current_timestep = 0
        running_reward = 0
        avg_length = 0

        self.best_score = -100 - 1 * self.config['Experiment']['loop_1']['max_timesteps']
        self.best_reward = self.best_score
        self.max_episodes = self.config['Experiment']['loop_1']['max_episodes']
        self.max_timesteps = self.config['Experiment']['loop_1']['max_timesteps']

        # self.test_agent(goal, 1)
        print("Continue Training.")

        for i_episode in range(1, self.max_episodes + 1):
            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            test_offline_score = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = True
            for timestep in range(1, self.max_timesteps + 1):
                test_game_start_time = time.time()
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                if not self.second_human:
                    randomness_threshold = self.config['Experiment']['loop_1']['stop_random_agent']
                    randomness_critirion = i_episode
                    flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)
                # compute keyboard action
                # duration_pause, _ = self.getKeyboard(actions, duration_pause)
                # get final action pair
                # action = self.get_action_pair()
                tmp_agent_action = self.agent_action
                if self.config["SAC"]["discrete"]:
                    tmp_agent_action = -1 if self.agent_action == abs(2) else self.agent_action

                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                observation_, reward, done, train_fps, duration_pause, action_list = self.env.step(tmp_agent_action, timedout, goal,
                                                           self.config['Experiment']['loop_1']['action_duration'], duration_pause)
                self.train_fps_list.append(train_fps)
                self.action_history = self.action_history + action_list
                # add experience to buffer
                interaction = [observation, self.agent_action, reward, observation_, done]
                self.save_experience(interaction)

                running_reward += reward
                episode_reward += reward
                test_offline_score += -1 if not done else 0

                # online train
                start_online_update = time.time()
                if not self.config['game']['test_model'] and not self.second_human:
                    if self.config['Experiment']['online_updates'] and i_episode >= self.config['Experiment']['loop_1'][
                        'start_training_step_on_episode']:
                        if self.discrete:
                            self.agent.learn()
                            self.agent.soft_update_target()
                self.online_update_duration_list.append(time.time() - start_online_update)

                observation = observation_
                # new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                #            "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                #            "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                #            "tray_rot_vel_y": observation[7]}
                new_row = {"ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df = self.df.append(new_row, ignore_index=True)
                # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
                #     sac.soft_update_target()
                test_step_duration = time.time() - test_game_start_time
                self.test_step_duration_list.append(test_step_duration)
                if done:
                    break

            end = time.time()
            if self.best_reward < episode_reward:
                self.best_reward = episode_reward
            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause

            self.episode_duration_list.append(episode_duration)
            self.score_history.append(episode_reward)

            log_interval = self.config['Experiment']['loop_1']['log_interval']
            avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
            avg_score = np.mean(self.score_history[-log_interval:])

            # best score logging
            # self.save_best_model(avg_score, i_episode, current_timestep)

            self.length_list.append(current_timestep)
            avg_length += current_timestep

            # if not self.config['Experiment']['online_updates']:
            #     self.test_score_history.append(self.config['Experiment']['test_loop']['max_score'] + test_offline_score)
            #     self.test_episode_duration_list.append(episode_duration)
            #     self.test_length_list.append(current_timestep)

            # off policy learning
            if not self.config['game']['test_model'] and i_episode >= self.config['Experiment']['loop_1'][
                'start_training_step_on_episode']:
                if i_episode % self.agent.update_interval == 0:
                    self.updates_scheduler()
                    if self.update_cycles > 0:
                        grad_updates_duration = self.grad_updates(self.update_cycles)
                        self.grad_updates_durations.append(grad_updates_duration)

                        # save the models after each grad update
                        self.agent.save_models()

                    # Test trials
                    if i_episode % self.config['Experiment']['test_interval'] == 0 and self.test_max_episodes > 0:
                        self.test_agent(goal)
                        print("Continue Training.")

            # logging
            if self.config["game"]["verbose"]:
                if not self.config['game']['test_model']:
                    running_reward, avg_length = self.print_logs(i_episode, running_reward, avg_length, log_interval,
                                                                 avg_ep_duration)
                current_timestep = 0
        update_cycles = math.ceil(
            self.config['Experiment']['loop_1']['total_update_cycles'])
        if not self.second_human and update_cycles > 0:
            try:
                self.avg_grad_updates_duration = mean(self.grad_updates_durations)
            except:
                print("Exception when calc grad_updates_durations")
        tracker.print_diff()

    # Experiment 2 loop
    def loop_2(self, goal):
        # Experiment 2 loop
        flag = True
        current_timestep = 0
        observation = self.env.reset()
        timedout = False
        episode_reward = 0
        actions = [0, 0, 0, 0]  # all keys not pressed

        self.best_score = -50 - 1 * self.config['Experiment']['loop_2']['max_timesteps_per_game']
        self.best_reward = self.best_score
        self.total_timesteps = self.config['Experiment']['loop_2']['total_timesteps']
        self.max_timesteps_per_game = self.config['Experiment']['loop_2']['max_timesteps_per_game']

        avg_length = 0
        duration_pause = 0
        self.save_models = True
        self.game = 0
        running_reward = 0
        start = time.time()

        for timestep in range(1, self.total_timesteps + 1):
            self.total_steps += 1
            current_timestep += 1

            # get agent's action
            if not self.second_human:
                randomness_threshold = self.config['Experiment']['loop_2']['start_training_step_on_timestep']
                randomness_critirion = timestep
                flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag)
            # compute keyboard action
            duration_pause, _ = self.getKeyboard(actions, duration_pause)
            # get final action pair
            action = self.get_action_pair()

            if current_timestep == self.max_timesteps_per_game:
                timedout = True

            # Environment step
            observation_, reward, done = self.env.step(action, timedout, goal,
                                                       self.config['Experiment']['loop_2']['action_duration'])

            interaction = [observation, self.agent_action, reward, observation_, done]
            # add experience to buffer
            self.save_experience(interaction)

            # online train
            if not self.config['game']['test_model'] and not self.second_human:
                if self.config['Experiment']['online_updates']:
                    if self.discrete:
                        self.agent.learn()
                        self.agent.soft_update_target()

            new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                       "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                       "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                       "tray_rot_vel_y": observation[7]}
            # append row to the dataframe
            self.df = self.df.append(new_row, ignore_index=True)
            observation = observation_

            # off policy learning
            if not self.config['game']['test_model'] and self.total_steps >= self.config['Experiment']['loop_2'][
                'start_training_step_on_timestep']:
                update_cycles = math.ceil(
                    self.config['Experiment']['loop_2']['update_cycles'])
                if self.total_steps % self.agent.update_interval == 0 and update_cycles > 0:
                    grad_updates_duration = self.grad_updates(update_cycles)
                    self.grad_updates_durations.append(grad_updates_duration)

                    # save the models after each grad update
                    self.agent.save_models()

                    # Test trials
                    if self.test_max_episodes > 0:
                        self.test_agent(goal)
                        print("Continue Training.")

            running_reward += reward
            episode_reward += reward

            if done:
                end = time.time()
                self.game += 1
                if self.best_reward < episode_reward:
                    self.best_reward = episode_reward
                self.duration_pause_total += duration_pause
                episode_duration = end - start - duration_pause

                self.episode_duration_list.append(episode_duration)
                self.score_history.append(episode_reward)

                log_interval = self.config['Experiment']['loop_2']['log_interval']
                avg_ep_duration = np.mean(self.episode_duration_list[-log_interval:])
                avg_score = np.mean(self.score_history[-log_interval:])

                # best score logging
                # self.save_best_model(avg_score, self.game, current_timestep)

                self.length_list.append(current_timestep)
                avg_length += current_timestep

                # logging
                if self.config["game"]["save"]:
                    if not self.config['game']['test_model']:
                        running_reward, avg_length = self.print_logs(self.game, running_reward, avg_length,
                                                                     log_interval,
                                                                     avg_ep_duration)

                current_timestep = 0
                observation = self.env.reset()
                timedout = False
                episode_reward = 0
                actions = [0, 0, 0, 0]  # all keys not pressed
                start = time.time()

        if not self.second_human:
            self.avg_grad_updates_duration = mean(self.grad_updates_durations)

    def test_human(self, goal):

        self.max_episodes = self.config['Experiment']['loop_1']['max_episodes']
        self.max_timesteps = self.config['Experiment']['loop_1']['max_timesteps']
        for i_episode in range(1, self.max_episodes + 1):
            self.env.reset()
            actions = [0, 0, 0, 0]  # all keys not pressed
            for step in range(self.max_timesteps):
                duration_pause, actions = self.getKeyboard(actions, 0)
                action = self.human_actions
                # Environment step
                observation_, reward, done = self.env.step(action, False, goal,
                                                           self.config['Experiment']['loop_1']['action_duration'])
                if done:
                    break

    def save_info(self, chkpt_dir, experiment_duration, total_games, goal):
        info = {}
        info['goal'] = goal
        info['experiment_duration'] = experiment_duration
        info['best_score'] = self.best_score
        info['best_score_episode'] = self.best_score_episode
        info['best_reward'] = self.best_reward
        info['best_score_length'] = self.best_score_length
        info['total_steps'] = self.total_steps
        info['total_games'] = total_games
        info['fps'] = self.env.fps
        info['avg_grad_updates_duration'] = self.avg_grad_updates_duration
        w = csv.writer(open(chkpt_dir + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

    def get_action_pair(self):
        if self.second_human:
            action = self.human_actions
        else:
            if self.config['game']['agent_only']:
                action = self.get_agent_only_action()
            else:
                tmp_agent_action = self.agent_action
                if self.config["SAC"]["discrete"]:
                    tmp_agent_action = -1 if self.agent_action == abs(2) else self.agent_action
                action = [tmp_agent_action, self.human_actions[1]]
        # self.action_history.append(action)
        return action

    def getKeyboard(self, actions, duration_pause):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return 1
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    # print("space")
                    start_pause = time.time()
                    pause()
                    end_pause = time.time()
                    duration_pause += end_pause - start_pause
                if event.key == pg.K_q:
                    exit(1)
                if event.key in self.env.keys:
                    actions[self.env.keys_fotis[event.key]] = 1
                    # action_human += maze.keys[event.key]
            if event.type == pg.KEYUP:
                if event.key in self.env.keys:
                    actions[self.env.keys_fotis[event.key]] = 0
                    # action_human -= maze.keys[event.key]
        self.human_actions = convert_actions(actions)

        if not self.config["SAC"]["discrete"]:
            # hit keyboard input with noise to make continuous
            # we use action as the mean of a normal distribution with variance 2
            self.human_actions = [random.gauss(self.human_actions[0], stdev), random.gauss(self.human_actions[1], stdev)]
        return duration_pause, actions

    def save_experience(self, interaction):
        observation, agent_action, reward, observation_, done = interaction
        if not self.second_human:
            if self.discrete:
                self.agent.memory.add(observation, agent_action, reward, observation_, done)
            else:
                self.agent.remember(observation, agent_action, reward, observation_, done)

    def save_best_model(self, avg_score, game, current_timestep):
        if avg_score > self.best_score:
            self.best_score = avg_score
            self.best_score_episode = game
            self.best_score_length = current_timestep
            if not self.config['game']['test_model'] and self.save_models and not self.second_human:
                self.agent.save_models()

    def grad_updates(self, update_cycles=None):
        start_grad_updates = time.time()
        end_grad_updates = 0
        if not self.second_human:
            print("Performing {} updates".format(update_cycles))
            for _ in tqdm(range(update_cycles)):
                if self.discrete:
                    self.agent.learn()
                    self.agent.soft_update_target()
                else:
                    self.agent.learn()
            end_grad_updates = time.time()

        return end_grad_updates - start_grad_updates

    def print_logs(self, game, running_reward, avg_length, log_interval, avg_ep_duration):
        if game % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            log_reward = int((running_reward / log_interval))

            print(
                'Episode {}\tTotal timesteps {}\tavg length: {}\tTotal reward(last {} episodes): {}\tBest Score: {}\tavg '
                'episode duration: {}'.format(game, self.total_steps, avg_length,
                                              log_interval,
                                              log_reward, self.best_score,
                                              timedelta(
                                                  seconds=avg_ep_duration)))
            running_reward = 0
            avg_length = 0
        return running_reward, avg_length

    def test_print_logs(self, avg_score, avg_length, best_score, duration):
        print(
            'Avg Score: {}\tAvg length: {}\tBest Score: {}\tTest duration: {}'.format(avg_score,
                                                                                      avg_length, best_score,
                                                                                      timedelta(seconds=duration)))

    def compute_agent_action(self, observation, randomness_critirion=None, randomness_threshold=None, flag=True):
        if self.discrete:
            if randomness_critirion is not None and randomness_threshold is not None \
                    and randomness_critirion <= randomness_threshold:
                # Pure exploration
                if self.config['game']['agent_only']:
                    self.agent_action = np.random.randint(pow(2, self.env.action_space.actions_number))
                else:
                    if self.discrete:
                        self.agent_action = np.random.randint(self.env.action_space.actions_number)
                    else:
                        self.agent_action = np.random.uniform(-1,1)
                self.save_models = False
                if flag:
                    print("Using Random Agent")
                    flag = False
            else:  # Explore with actions_prob
                self.save_models = True
                self.agent_action = self.agent.actor.sample_act(observation)
                if not flag:
                    print("Using SAC Agent")
                    flag = True
        else:
            self.save_models = True
            self.agent_action = self.agent.choose_action(observation)
        return flag

    def test_agent(self, goal, randomness_critirion=None):
        # test loop
        current_timestep = 0
        self.test += 1
        print('Test {}'.format(self.test))
        best_score = 0
        flag = True
        for game in range(1, self.test_max_episodes + 1):
            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            for timestep in range(1, self.test_max_timesteps + 1):
                current_timestep += 1
                # compute agent's action
                randomness_threshold = self.config['Experiment']['loop_2']['start_training_step_on_timestep']
                flag = self.compute_agent_action(observation, randomness_critirion, randomness_threshold, flag=flag)
                # compute keyboard action
                # duration_pause, _ = self.getKeyboard(actions, duration_pause)
                # get final action pair
                # action = self.get_action_pair()
                tmp_agent_action = self.agent_action
                if self.config["SAC"]["discrete"]:
                    tmp_agent_action = -1 if self.agent_action == abs(2) else self.agent_action

                if timestep == self.test_max_timesteps:
                    timedout = True

                # Environment step
                observation_, _, done, test_fps, duration_pause, action_list = self.env.step(tmp_agent_action, timedout, goal,
                                                      self.config['Experiment']['test_loop']['action_duration'], duration_pause)
                self.test_fps_list.append(test_fps)
                self.action_history = self.action_history + action_list

                observation = observation_
                # new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                #            "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                #            "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                #            "tray_rot_vel_y": observation[7]}
                new_row = {"ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                episode_reward += -1

                if done:
                    break

            end = time.time()

            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause
            episode_score = self.config['Experiment']['test_loop']['max_score'] + episode_reward

            self.test_episode_duration_list.append(episode_duration)
            self.test_score_history.append(episode_score)
            self.test_length_list.append(current_timestep)
            best_score = episode_score if episode_score > best_score else best_score

            current_timestep = 0

        # logging
        self.test_print_logs(mean(self.test_score_history[-10:]), mean(self.test_length_list[-10:]), best_score,
                             sum(self.test_episode_duration_list[-10:]))

    def get_agent_only_action(self):
        # up: 0, down:1, left:2, right:3, upleft:4, upright:5, downleft: 6, downright:7
        if self.agent_action == 0:
            return [1, 0]
        elif self.agent_action == 1:
            return [-1, 0]
        elif self.agent_action == 2:
            return [0, -1]
        elif self.agent_action == 3:
            return [0, 1]
        elif self.agent_action == 4:
            return [1, -1]
        elif self.agent_action == 5:
            return [1, 1]
        elif self.agent_action == 6:
            return [-1, -1]
        elif self.agent_action == 7:
            return [-1, 1]
        else:
            print("Invalid agent action")

    def test_loop(self):
        # test loop
        current_timestep = 0
        self.test += 1
        print('Test {}'.format(self.test))
        goals = [left_down, right_down, left_up, ]
        for game in range(1, self.test_max_episodes + 1):
            # randomly choose a goal
            current_goal = random.choice(goals)

            observation = self.env.reset()
            timedout = False
            episode_reward = 0
            start = time.time()

            actions = [0, 0, 0, 0]  # all keys not pressed
            duration_pause = 0
            self.save_models = False
            for timestep in range(1, self.test_max_timesteps + 1):
                self.total_steps += 1
                current_timestep += 1

                # compute agent's action
                self.compute_agent_action(observation)
                # compute keyboard action
                duration_pause, _ = self.getKeyboard(actions, duration_pause)
                # get final action pair
                action = self.get_action_pair()

                if timestep == self.max_timesteps:
                    timedout = True

                # Environment step
                observation_, reward, done = self.env.step(action, timedout, current_goal,
                                                           self.config['Experiment']['test_loop']['action_duration'])

                observation = observation_
                new_row = {'actions_x': action[0], 'actions_y': action[1], "ball_pos_x": observation[0],
                           "ball_pos_y": observation[1], "ball_vel_x": observation[2], "ball_vel_y": observation[3],
                           "tray_rot_x": observation[4], "tray_rot_y": observation[5], "tray_rot_vel_x": observation[6],
                           "tray_rot_vel_y": observation[7]}
                # append row to the dataframe
                self.df_test = self.df_test.append(new_row, ignore_index=True)

                episode_reward += reward

                if done:
                    break

            end = time.time()

            self.duration_pause_total += duration_pause
            episode_duration = end - start - duration_pause

            self.test_episode_duration_list.append(episode_duration)
            self.test_score_history.append(self.config['Experiment']['test_loop']['max_score'] + episode_reward)
            self.test_length_list.append(current_timestep)

            # logging
            # self.test_print_logs(game, episode_reward, current_timestep, episode_duration)

            current_timestep = 0

    def updates_scheduler(self):
        update_list = [22000, 1000, 1000, 1000, 1000, 1000, 1000]
        total_update_cycles = self.config['Experiment']['loop_1']['total_update_cycles']
        online_updates = 0
        if self.config['Experiment']['online_updates']:
            online_updates = self.max_timesteps * (
                    self.max_episodes - self.config['Experiment']['loop_1']['start_training_step_on_episode'])

        if self.update_cycles is None:
            self.update_cycles = total_update_cycles - online_updates

        if self.config['Experiment']['scheduling'] == "descending":
            self.counter += 1
            if not (math.ceil(self.max_episodes / self.agent.update_interval) == self.counter):
                self.update_cycles /= 2

        elif self.config['Experiment']['scheduling'] == "big_first":
            if self.config['Experiment']['online_updates']:
                if self.counter == 1:
                    self.update_cycles = update_list[self.counter]
                else:
                    self.update_cycles = 0
            else:
                self.update_cycles = update_list[self.counter]

            self.counter += 1

        else:
            self.update_cycles = (total_update_cycles - online_updates) / math.ceil(
                self.max_episodes / self.agent.update_interval)

        self.update_cycles = math.ceil(self.update_cycles)
