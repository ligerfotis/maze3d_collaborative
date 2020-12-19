right_down = [104, 104]
# center = [-24, 8]
center = [-22, 9]


def checkTerminal(ball):
    # print("Ball x:{} y{}".format(ball.x, ball.y))
    if ball.x < - center[0] and ball.y < center[1]:
        return True
    else:
        return False
