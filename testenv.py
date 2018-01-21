from env4 import TicTacToeEnv as env
import gym
x=env()
x.step(0)
x.step(1)
x.step(4)
x.step(2)
x.step(8)
x.show_board()
