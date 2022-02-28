import numpy as np


# The squares on board used is (25 mm x 25 mm)
# cam0 : stereo_left
# cam1 : stereo_right
# cam2 : ir
# cam3 : ir_padded
# cam4 : ir_upsampled

def create_board(n0, n1):
    # creates a board of n0xn1
    ch_board = np.zeros((n0 * n1, 3), np.float32)
    ch_board[:, :2] = np.mgrid[0:n0, 0:n1].T.reshape(-1, 2)
    return ch_board


# some constants for this case
img_size = {0: (720, 1280), 1: (720, 1280), 2: (171, 224), 3: (720, 1280), 4: (720, 1280)}

board_size = (6, 9)
board = create_board(*board_size)
