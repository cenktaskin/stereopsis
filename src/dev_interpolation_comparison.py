import cv2
import torch
import torchvision.transforms
from torch.utils.data import DataLoader

import math
from dataset import StereopsisDataset, data_path
import matplotlib.pyplot as plt

def show_comparison_results(imgs, titles, col_count=2):
    row_count = math.ceil(len(imgs) / col_count)
    titles = ["original"] + titles
    for i, img in enumerate(imgs):
        plt.subplot(row_count, col_count, i + 1)
        try:
            plt.title(f"{titles[i]} - max:{img.max():.4f}")
        except:
            pass
        plt.imshow(img, interpolation=None)
    plt.subplots_adjust(hspace=0.25)
    plt.show()


dataset_id = "20220610"
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-origres")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
data_split_ratio = 0.98
dataset = StereopsisDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

sample_x, sample_y = next(iter(dataloader))

new_size = [384, 768]

cv_sample = sample_y.squeeze().numpy()
cv_modes = [cv2.INTER_LINEAR, cv2.INTER_LINEAR_EXACT]

cv_res = [cv2.resize(cv_sample, dsize=new_size[::-1], interpolation=m) for m in cv_modes]
cv_res = [cv_sample] + cv_res

scale_count = 6
avg_layers = [torch.nn.AvgPool2d(kernel_size=2 ** (i + 1)) for i in range(scale_count)]
max_layers = [torch.nn.MaxPool2d(kernel_size=2 ** (i + 1)) for i in range(scale_count)]

to_tens = torchvision.transforms.ToTensor()
tit = ["bilinear", "nearest-exact"]
res = []
for s in avg_layers:
    for img in cv_res:
        res += s(to_tens(img))
        print(res[-1].shape)

show_comparison_results(cv_res + res, tit, col_count=3)
