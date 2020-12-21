right_down = [104, 104]
# center = [-24, 8]
center = [-22, 9]


def checkTerminal(ball):
    # print("Ball x:{} y{}".format(ball.x, ball.y))
    if ball.x < - center[0] and ball.y < center[1]:
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
