from PIL import Image
from cv2 import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import numpy as np
from dataset import data_path

img_path = data_path.joinpath("raw/dataset-20220610")
ts = np.random.choice(list(img_path.glob("sl*"))).stem[3:]
img_l = Image.open(img_path.joinpath(f"sl_{ts}.tiff"))
img_r = Image.open(img_path.joinpath(f"sr_{ts}.tiff"))
img_d = np.array(Image.open(img_path.joinpath(f"dp_{ts}.tiff")))

# pool = torch.nn.AvgPool2d(kernel_size=3)
# output = pool(img_d)

img_d2 = cv2.imread(img_path.joinpath(f"dp_{ts}.tiff").as_posix(), flags=cv2.IMREAD_UNCHANGED)
print(img_d.max())
print(img_d.min())
print(img_d2.shape)
print(img_d2.dtype)
print(img_d2.max())
print(img_d2.min())
tensor_img = torch.from_numpy(img_d2).unsqueeze(dim=0).unsqueeze(dim=0)
print(tensor_img.shape)
print(tensor_img.dtype)
print(tensor_img.max())
print(tensor_img.min())

plt.subplot(1, 2, 1)
plt.title("pillow")
plt.imshow(img_d)
plt.subplot(1, 2, 2)
plt.title("cv")
plt.imshow(img_d2)
plt.show()
exit()

new_size = [11, 20]
depth_interpolated = torch.nn.functional.interpolate(tensor_img, size=new_size, mode="nearest")
print(depth_interpolated.dtype)
print(depth_interpolated.max())
print(depth_interpolated.min())

depth_interpolated2 = torch.nn.functional.interpolate(tensor_img, size=new_size, mode="nearest-exact")  # winner
print(depth_interpolated2.dtype)
print(depth_interpolated2.max())
print(depth_interpolated2.min())

depth_interpolated3 = torch.nn.functional.interpolate(tensor_img, size=new_size, mode="bilinear")
print(depth_interpolated3.dtype)
print(depth_interpolated3.max())
print(depth_interpolated3.min())

plt.subplot(2, 2, 1)
plt.title("original")
plt.imshow(img_d)
plt.subplot(2, 2, 2)
plt.title("nearest")
plt.imshow(depth_interpolated.squeeze())
plt.subplot(2, 2, 3)
plt.title("nearest-exact")
plt.imshow(depth_interpolated2.squeeze())
plt.subplot(2, 2, 4)
plt.title("bilinear")
plt.imshow(depth_interpolated3.squeeze())
plt.show()

# loader = torch.load('val_loader.pth')
# print(loader.dataset.indices)
