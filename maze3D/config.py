import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
import numpy as np
import pyrr
import pywavefront as pwf
from maze3D.assets import *

gameDisplay = None
display_width, display_height = [800, 800]
pg.font.init()
font = pg.font.SysFont("Grobold", 20)  # Assign it to a variable font


def text_objects(text, color):
    textSurface = font.render(text, True, color)
    return textSurface, textSurface.get_rect()


def message_to_screen(msg, color, param1, size):
    textSurf, textRect = text_objects(msg, color)
    textRect.center = (display_width / 2), (display_height / 2)
    gameDisplay.blit(textSurf, textRect)


def pause():
    pause = True
    while pause:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    pause = False

        # message_to_screen("Paused", BLACK, -100, size="large")
        # gameDisplay.update()
        # clock.tick(5)


pg.init()

pg.display.set_mode((display_width, display_height), pg.OPENGL | pg.DOUBLEBUF)
clock = pg.time.Clock()

glClearColor(0, 0.0, 0.0, 1)

with open("maze3D/shaders/vertex.txt", 'r') as f:
    vertex_src = f.readlines()
with open("maze3D/shaders/fragment.txt", 'r') as f:
    fragment_src = f.readlines()
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                        compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

# get a handle to the rotation matrix from the shader
MODEL_LOC = glGetUniformLocation(shader, "model")
VIEW_LOC = glGetUniformLocation(shader, "view")
PROJ_LOC = glGetUniformLocation(shader, "projection")
LIGHT_LOC = glGetUniformLocation(shader, "lightPos")

glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glEnable(GL_CULL_FACE)

########################MODELS######################################
BOARD_MODEL = ObjModel("maze3D/models/board.obj")
WALL_MODEL = ObjModel("maze3D/models/wall.obj")
BALL_MODEL = ObjModel("maze3D/models/ball.obj")
########################TEXTURES####################################
BOARD = Texture("maze3D/textures/board.jpg")
WALL = Texture("maze3D/textures/wall.jpg")
BALL = Texture("maze3D/textures/glass.png")
####################################################################

# (field of view, aspect ratio,near,far)
cameraPos = pyrr.Vector3([0, -450, 500])
up = pyrr.Vector3([0.0, 0.0, 1.0])
cameraRight = pyrr.vector.normalise(pyrr.vector3.cross(up, cameraPos))
cameraUp = pyrr.vector3.cross(cameraPos, cameraRight)
viewMatrix = pyrr.matrix44.create_look_at(cameraPos, pyrr.Vector3([0, 0, 0]), cameraUp)
# projection = pyrr.matrix44.create_perspective_projection_matrix(45, 600 / 600, 320, 720)
# Fotis: Changed far clipping plane
projection = pyrr.matrix44.create_perspective_projection_matrix(45, 800 / 800, 320, 1000)
glUniformMatrix4fv(PROJ_LOC, 1, GL_FALSE, projection)
glUniformMatrix4fv(VIEW_LOC, 1, GL_FALSE, viewMatrix)

lightPosition = pyrr.Vector3([-400.0, 200.0, 500.0])
glUniform3f(LIGHT_LOC, -400.0, 200.0, 300.0)

### EFFORT TO ADD A WELL####
vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )
edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )
def Cube():
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()