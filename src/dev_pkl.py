import pickle as pkl
from pathlib import Path
from cv2 import cv2

with open(Path.cwd().parent.joinpath("undistorted.pickle"), 'rb') as f:
    pkl_img = pkl.load(f)

for item in pkl_img:
    print(f"{type(item)}")
    for subitem in item:
        print(f"{type(subitem)}")
        cv2.imshow('ss',subitem)
        cv2.waitKey()

