import numpy as np


# cam0 : stereo_left
# cam1 : stereo_right
# cam2 : ir
# cam3 : ir_upsampled

def create_board(n0, n1, square_size=1):
    # creates a board of n0xn1
    ch_board = np.zeros((n0 * n1, 3), np.float32)
    ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
    return ch_board * square_size


# some constants for this case
img_size = {0: (720, 1280), 1: (720, 1280), 2: (171, 224), 3: (720, 1280), 4: (720, 1280)}

square_size = 25.5 # in mm
board_size = (6, 9)
board = create_board(*board_size, square_size)
