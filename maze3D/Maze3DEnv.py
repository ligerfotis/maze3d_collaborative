import random
import time
from maze3D.gameObjects import *
from maze3D.assets import *
from maze3D.utils import checkTerminal, get_distance_from_goal, checkTerminal_new, convert_actions
from rl_models.utils import get_config
from maze3D.config import layout_up_right, layout_down_right, layout_up_left

layouts = [layout_down_right, layout_up_left, layout_up_right]


class ActionSpace:
    def __init__(self):
        # self.actions = list(range(0, 14 + 1))
        # self.shape = 1
        self.actions = list(range(0, 3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        # return [random.sample([0, 1, 2], 1), random.sample([0, 1, 2], 1)]
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    def __init__(self, config=None, config_file=None):
        self.fps_list = []
        # choose randomly one starting point for the ball
        current_layout = random.choice(layouts)
        # current_layout = layout_up_right
        self.board = GameBoard(current_layout)
        self.keys = {pg.K_UP: 1, pg.K_DOWN: 2, pg.K_LEFT: 4, pg.K_RIGHT: 8}
        self.keys_fotis = {pg.K_UP: 0, pg.K_DOWN: 1, pg.K_LEFT: 2, pg.K_RIGHT: 3}
        self.running = True
        self.done = False
        self.observation = self.get_state()  # must init board fisrt
        self.action_space = ActionSpace()
        self.observation_shape = (len(self.observation),)
        self.dt = None
        self.fps = 60
        self.config = get_config(config_file) if config_file is not None else config
        self.reward_type = self.config['SAC']['reward_function'] if 'SAC' in self.config.keys() else None
        self.discrete = self.config['SAC']['discrete']

    def step(self, action_agent, timedout, goal, action_duration, duration_pause):
        tmp_time = time.time()
        actions = [0, 0, 0, 0]
        action_list = []
        while (time.time() - tmp_time) < action_duration and not self.done:
            # print("Env got action: {}".format(action))
            # self.board.handleKeys(action)  # action is int
            # compute keyboard action
            duration_pause, _ = self.getKeyboard(actions, duration_pause)
            action = [action_agent, self.human_actions[1]]
            action_list.append(action)

            self.board.handleKeys_fotis(action)
            self.board.update()
            glClearDepth(1000.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.board.draw()
            pg.display.flip()

            self.dt = clock.tick(self.fps)
            fps = clock.get_fps()
            self.fps_list.append(fps)

            pg.display.set_caption("Running at " + str(int(fps)) + " fps")
            self.observation = self.get_state()
            if checkTerminal_new(self.board.ball, goal) or timedout:
                self.done = True
        reward = self.reward_function_maze(timedout, goal)
        return self.observation, reward, self.done, fps, duration_pause, action_list

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
                if event.key in self.keys:
                    actions[self.keys_fotis[event.key]] = 1
                    # action_human += maze.keys[event.key]
            if event.type == pg.KEYUP:
                if event.key in self.keys:
                    actions[self.keys_fotis[event.key]] = 0
                    # action_human -= maze.keys[event.key]
        self.human_actions = convert_actions(actions)
        return duration_pause, actions


    def get_state(self):
        # [ball pos x | ball pos y | ball vel x | ball vel y|  theta(x) | phi(y) |  theta_dot(x) | phi_dot(y) | ]
        return np.asarray(
            [self.board.ball.x, self.board.ball.y, self.board.ball.velocity[0], self.board.ball.velocity[1],
             self.board.rot_x, self.board.rot_y, self.board.velocity[0], self.board.velocity[1]])

    def reset(self):
        self.__init__(config=self.config)
        return self.observation

    def reward_function_maze(self, timedout, goal=None):
        if self.reward_type == "Sparse" or self.reward_type == "sparse":
            return self.reward_function_sparse(timedout)
        elif self.reward_type == "Dense" or self.reward_type == "dense":
            return self.reward_function_dense(timedout, goal)
        elif self.reward_type == "Sparse_2" or self.reward_type == "sparse_2":
            return self.reward_function_sparse2(timedout)

    def reward_function_sparse(self, timedout):
        # For every timestep -1
        # Timed out -50
        # Reach goal +100
        if self.done and not timedout:
            # solved
            return 100
        # if not done and timedout
        if timedout:
            return -50
        # return -1 for each time step
        return -1

    def reward_function_dense(self, timedout, goal=None):
        # For every timestep -target_distance
        # Timed out -50
        # Reach goal +100
        if self.done:
            # solved
            return 100
        # if not done and timedout
        if timedout:
            return -50
        # return -target_distance/10 for each time step
        target_distance = get_distance_from_goal(self.board.ball, goal)
        return -target_distance / 10

    def reward_function_sparse2(self, timedout):
        # For every timestep -1
        # Timed out -50
        # Reach goal +100
        if self.done and not timedout:
            # solved
            return 10
        # return -1 for each time step
        return -1
