import cv2
import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from dataset import StereopsisDataset, data_path, show_images

dataset_id = "20220610"
dataset_path = data_path.joinpath(f"processed/dataset-{dataset_id}-origres")
current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1
data_split_ratio = 0.98
dataset = StereopsisDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

sample_x, sample_y = next(iter(dataloader))

print(sample_y.shape)
new_size = [384, 768]

cv_sample = sample_y.squeeze().numpy()
cv_res = cv2.resize(cv_sample, dsize=new_size[::-1], interpolation=cv2.INTER_LINEAR_EXACT)

scale_count = 6
avg_layers = [torch.nn.AvgPool2d(kernel_size=2 ** i) for i in range(scale_count, 0, -1)]

to_tens = torchvision.transforms.ToTensor()
tit = ["orig", "original_upsampled"]
res = [cv_sample, cv_res]

for i, avg_lay in enumerate(avg_layers):
    avg_res = avg_lay(to_tens(cv_res))
    res += [avg_res]
    tit += [f"avg"]
    print(f"{cv_res.shape}->{res[-1].shape}")

    back = cv2.resize(avg_res.squeeze().numpy(), dsize=new_size[::-1], interpolation=cv2.INTER_LINEAR_EXACT)
    res += [back]
    tit += [f"back"]

show_images(res, tit, row_count=7)
