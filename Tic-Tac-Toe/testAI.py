from env4 import TicTacToeEnv as env
from AI import AI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def csv_to_arr(file):
 df=pd.read_csv(file,sep=',',header=None)
 return(df.values)
env=env()
player1=AI(env,1)
player2=AI(env,2)
player1.q=csv_to_arr('data2.csv')
player2.q=csv_to_arr('data1.csv')
done=False
win=0
loss=0
draw=0
list1=[]
list2=[]
for i in range(10000):
 env.reset()
 player1.reset()
 player2.reset()
 done=False
 reward=0
 while (done==False):
  reward,done=player1.nextState()
  if (reward==1):
   win=win+1
  if (reward==-1):
   loss=loss+1
  if ((reward==0) & (done==True)):
   draw=draw+1
  list1=list1+[win-loss]
  list2=list2+[i]
  #env.show_board()
  if (done):
   break
  reward,done=player2.nextState()
  if ((reward==0) & (done==True)):
   draw=draw+1
print('win=',win)
print('loss=',loss)
print('draw=',draw)
plt.plot(list2,list1)
plt.show()