from env4 import TicTacToeEnv as env
from AI import AI
import gym
import numpy as np
import pandas as pd
def arr_to_csv(x,file):
 np.savetxt(file,x,delimiter=',')
def csv_to_arr(file):
 df=pd.read_csv(file,sep=',',header=None)
 return(df.values)
reward=0
done=False
env=env()
player1=AI(env,1)
player2=AI(env,2)
for i in range(1000000):
 #print(i)
 env.reset()
 player1.reset()
 player2.reset()
 done=False
 while (done==False):
  reward,done=player1.nextState()
  #env.show_board()
  #print(reward)
  if (done):
   break
  reward,done=player2.nextState()
  #env.show_board()
  #print(reward)
arr_to_csv(player1.q,'data1.csv')
arr_to_csv(player2.q,'data2.csv')