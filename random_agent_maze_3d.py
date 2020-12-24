from maze3D.Maze3DEnv import Maze3D
from maze3D.assets import *


def main():
    maze = Maze3D()
    action = 0
    observation = maze.reset()
    while 1:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                maze.running = False
            # if event.type == pg.KEYDOWN:
            #     if event.key in maze.keys:
            #         action += maze.keys[event.key]
            # if event.type == pg.KEYUP:
            #     if event.key in maze.keys:
            #         action -= maze.keys[event.key]
        action = maze.action_space.sample()
        observation_, done = maze.step(action)
        if done:
            break

    pg.quit()


if __name__ == '__main__':
    main()
    exit(0)
