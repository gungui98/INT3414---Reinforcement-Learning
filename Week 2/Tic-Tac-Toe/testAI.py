import pandas as pd

from AI import AI
from env4 import TicTacToeEnv as env


def csv_to_arr(file):
    df = pd.read_csv(file, sep=',', header=None)
    return df.values


env = env()
player1 = AI(env, 2)
player2 = AI(env, 1)
player1.q = csv_to_arr('data1.csv')
player2.q = csv_to_arr('data1.csv')
p1 = 0
p2 = 0
for i in range(1000):
    env.reset()
    player1.reset()
    player2.reset()
    while not env.done:
        p1 = p1 + player1.nextState()
        # env.show_board()
        if env.done:
            break
        p2 = p2 + player2.nextState()
    env.reset()
    player1.reset()
    player2.reset()
    while not env.done:
        p1 = p1 + player2.nextState()
        # env.show_board()
        if env.done:
            break
        p2 = p2 + player1.nextState()
print(p1, p2)
