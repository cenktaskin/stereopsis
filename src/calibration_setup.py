import numpy as np


def create_board(n0, n1, square_size=1):
    # creates a board of n0xn1
    ch_board = np.zeros((n0 * n1, 3), np.float32)
    ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
    return ch_board * square_size


square_size = 25.5  # in mm
board_size = (6, 9)
board = create_board(*board_size, square_size)
