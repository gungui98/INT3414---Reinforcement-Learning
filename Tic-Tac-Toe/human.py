from env4 import TicTacToeEnv as env
from AI import AI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def csv_to_arr(file):
 df=pd.read_csv(file,sep=',',header=None)
 return(df.values)
env=env()
player2=AI(env,2)
player2.q=csv_to_arr('data1.csv')
done=False
env.reset()
player2.reset()
done=False
reward=0
tick=0
while (True):
  tick=int(input())
  env.step(tick)
  env.show_board()
  if (env.status!=-1):
   print('Win') if (env.status==1) else print('Draw')
   break
  player2.nextState()
  env.show_board()
  if (env.status!=-1):
   print('Loss') if (env.status==2) else print('Draw')
   break