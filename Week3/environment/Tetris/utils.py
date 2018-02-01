import numpy as np
"""
this is collection of helpser class needed for running tetris

"""
STICK_STR = "0 0	0 1	 0 2  0 3"
L1_STR = "0 0	0 1	 0 2  1 0"
L2_STR = "0 0	1 0 1 1	 1 2"
S1_STR = "0 0	1 0	 1 1  2 1"
S2_STR = "0 1	1 1	 1 0  2 0"
SQUARE_STR = "0 0	0 1	 1 0  1 1"
PYRAMID_STR = "0 0  1 0  1 1	2 0"

STICK = 0
L1 = 1
L2 = 2
S1 = 3
S2 = 4
SQUARE = 5
PYRAMID = 6

class Tpoint:
    """
    point state
    """
    def __init__(self,x ,y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return other.x == self.x && other.y == self.y


class Piece:
    """
    class for tetris piece
    """
    def __init__(self,points):
        """
        initialize object
        :param points: a string or list of point
        """
        if type(points) == str:
            self.parse_point(points)
        else:
            self.body = points

        self.height = 0
        self.width = 0
        self.skirt = np.ones(len(points))


        for i in  enumerate(self.body):
            if i.x >= self.width :
                self.width = i.x
            if i.y >= self.height:
                self.height = i.y

        self.skirt *= self.height
        for x in range(self.width) :
            for i in enumerate(self.body):
                if i.x == x and i.y < self.skirt[x]:
                    self.skirt[x] = i.y

        self.next = None
        self.pieces = []

    def parse_point(self,points):
        """
        convert string to list of points
        :param points a string contains information about point
        :return:list of coordinate of points
        """
        self.body = None

    def compute_next(self):
        """
        compute next piece from current piece
        :return: piece  next piece
        """
        tpoints = []
        for i in range(len(self.body)):
            x = self.height - self.body[i].y - 1;
            y = self.body[i].x;
            tpoint = Tpoint(x,y)
            tpoints.append(tpoint)
        return Piece(tpoints);

    def fast_rolate(self):
        """
        compute fast relation of coordinate
        :return:
        """
        return self.next

    def __eq__(self, other):
        """
        compare to other piece
        :param other: other piece
        :return: whether 2 piece is the same
        """
        if type(self) is not type(other):
            return False
        if self is other:
            return True
        if len(self.body) != len(other.body):
            return False
        for i in enumerate(len(self.body)):
            if not self.body[i] == other.body[i]:
                return False
        return True

    def get_pieces(self):
        """
        return list of possible piece from current piece
        :return: list of pieces
        """
        if len(self.pieces) == 0:
            self.pieces = [self.make_fast_rolate(Piece(STICK_STR)),
                           self.make_fast_rolate(Piece(L1_STR)),
                           self.make_fast_rolate(Piece(L2_STR)),
                           self.make_fast_rolate(Piece(S1_STR)),
                           self.make_fast_rolate(Piece(S2_STR)),
                           self.make_fast_rolate(Piece(SQUARE_STR)),
                           self.make_fast_rolate(Piece(PYRAMID_STR))]

    def make_fast_rolate(self,root):


