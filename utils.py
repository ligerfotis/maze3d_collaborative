# right_down = [104, 104]
# center = [-24, 8]
center = [0, 0]
left_down = [-104, -104]
left_up = [-104, 73]
right_down = [73, -104]

#################
goal = left_up
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

