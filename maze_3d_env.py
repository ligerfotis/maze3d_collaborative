import random

from config import *
from gameObjects import *
from assets import *
from utils import checkTerminal

layout = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
          [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
          [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
          [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
          [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
          [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
          [1, 1, 0, 0, 0, 0, 0, 0, 2, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


class Maze3D:
    def __init__(self):
        self.board = GameBoard(layout)
        self.keys = {pg.K_UP: 1, pg.K_DOWN: 2, pg.K_LEFT: 4, pg.K_RIGHT: 8}
        self.running = True
        self.done = False
        self.observation = self.get_state() # must init board fisrt
        self.action_space = []

    def step(self, action):
        self.board.handleKeys(action)  # action is int
        self.board.update()
        glClearDepth(1000.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.board.draw()
        pg.display.flip()
        clock.tick()
        fps = clock.get_fps()
        pg.display.set_caption("Running at " + str(int(fps)) + " fps")
        self.observation = self.get_state()
        if checkTerminal(self.board.ball):
            self.done = True
        return self.observation, self.done

    def get_state(self):
        # [ball pos x | ball pos y | ball vel x | ball vel y|  theta(x) | phi(y) ]
        return self.board.ball.x, self.board.ball.y, self.board.ball.velocity[0], self.board.ball.velocity[1], \
               self.board.rot_x, self.board.rot_y

    def reset(self):
        self.__init__()
        return self.observation

    class action_space:
        def __init__(self):
            pass
            self.actions = [1, 2, 4, 8]

        def sample(self):
            random.sample(self.actions, 5)

def main():
    maze = Maze3D()
    action = 0
    observation = maze.reset()
    done = False
    while not done:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                maze.running = False
            if event.type == pg.KEYDOWN:
                if event.key in maze.keys:
                    action += maze.keys[event.key]
            if event.type == pg.KEYUP:
                if event.key in maze.keys:
                    action -= maze.keys[event.key]
        observation_, done = maze.step(action)
        # if done:
        #     maze.reset()

    pg.quit()


if __name__ == '__main__':
    main()
    exit(0)
