import pickle
import numpy as np
from pathlib import Path
from PIL import Image

data_path = Path('/home/cenkt/Downloads/undistorted.pickle')
data = pickle.load(open(data_path, 'rb'))

for i, example in enumerate(data):
    normalized_label = np.round((example[1]-example[1].min())/(example[1].max()-example[1].min()) * 255)
    Image.fromarray(normalized_label).convert('L').save(f"dataset/dist/{i:05}.png")
    Image.fromarray(example[0]).save(f"dataset/image/{i:05}.png")
