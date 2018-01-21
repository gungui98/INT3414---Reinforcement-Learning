CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
SIZE=3
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0
def tomark(code):
    return CODE_MARK_MAP[code]
def tocode(mark):
    return 1 if mark == 'O' else 2
def next_mark(mark):
    return 'X' if mark == 'O' else 'O'
def check_game_status(board):
    """Return game status by current board status.
    Args:
        board (list): Current board state
    Returns:
        int:
            -1: game in progress
            0: draw game,
            1 or 2 for finished game(winner mark code).
    """
    for t in [1, 2]:
        for j in range(0, SIZE*SIZE, SIZE):
            if [t] * SIZE == [board[i] for i in range(j, j+SIZE)]:
                return t
        for j in range(0, SIZE):
            if [t]* SIZE == [board[j+i*SIZE] for i in range(0,SIZE)]:
                return t
        if [t]*SIZE==[board[i] for i in range(0,SIZE*SIZE,SIZE+1)]: 
         return t
        if [t]*SIZE==[board[i] for i in range(SIZE-1,SIZE*(SIZE-1)+1,SIZE-1)]:
         return t
 

    for i in range(SIZE*SIZE):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0
class TicTacToeEnv():
 def __init__(self):
  self.board=[0]*SIZE*SIZE
  self.start_mark='X'
  self.reset()
 def reset(self):
  self.board = [0] * SIZE * SIZE
  self.mark = self.start_mark
  self.done=False
 def step(self,tick): # tick is location to tick
  if (self.done==True):
   return
  reward=NO_REWARD
  self.board[tick]=tocode(self.mark)
  status=check_game_status(self.board)
  if status >= 0:
   self.done = True
   if status in [1, 2]:
    # always called by self
    reward = O_REWARD if self.mark == 'O' else X_REWARD
    print('O win') if status==1 else print('X win')
   if status==0:
    print('Draw game')
   return
  self.mark=next_mark(self.mark)
 def show_board(self):
  for i in range(0,SIZE):
   print([tomark(self.board[j+i*SIZE]) for j in range(0,SIZE)])
  print()
 
   
 
 
   
	
  
  
  
	
