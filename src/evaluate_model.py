from models import BeelineModel
from dataset import *
import torch
from torch.utils.data import Subset
from preprocessing.data_io import data_path
from loss import MaskedMSE
import tarfile

device = "cuda" if torch.cuda.is_available() else "cpu"

train_id = "202205011633"
dataset_id = "20220301"
model = BeelineModel()

with tarfile.open(f"/home/cenkt/downloads/{train_id}.tar") as f:
    model_file = f.extractfile("model.pth")
    model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    idx_file = f.extractfile("test_indices.txt")
    test_idx = list(np.loadtxt(idx_file).astype(int))


dataset_path = data_path.joinpath(f"raw/data-{dataset_id}")
label_transformer = LabelTransformer(h=120, w=214)
dataset = StereopsisDataset(dataset_path, transform=transforms.Compose([np_to_tensor]),
                            target_transform=transforms.Compose([label_transformer]))



test_dataset = Subset(dataset, test_idx)
print(test_dataset.indices)
BATCH_SIZE = 32
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = MaskedMSE()

output_dir = Path("/content/drive/MyDrive/stereopsis/some_test_results/")
with torch.no_grad():
    sample = next(iter(test_dataloader))
    sample_idx = np.random.randint(0, high=sample[0].shape[0], size=5)
    x_l_batch, x_r_batch, y_batch = sample[0].to(device), sample[1].to(device), sample[2].to(device)
    pred_batch = model(x_l_batch, x_r_batch)
    test_loss = loss_fn(pred_batch, y_batch).item()
    for idx in sample_idx:
        x_l, x_r, y, pred = x_l_batch[idx].cpu(), x_r_batch[idx].cpu(), y_batch[idx].cpu(), pred_batch[idx].cpu()
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle(f'Test loss {test_loss:.4f}')
        plt.subplot(2, 2, 1)
        plt.imshow(x_l.permute(1, 2, 0))  # back to np dimension order
        plt.title('Left input image')
        plt.subplot(2, 2, 2)
        plt.imshow(x_r.permute(1, 2, 0))  # back to np dimension order
        plt.title('Right input image')
        plt.subplot(2, 2, 3)
        plt.imshow(y)
        plt.title('Label')
        plt.subplot(2, 2, 4)
        plt.imshow(pred.squeeze())
        plt.title('Predicted')
        plt.show()
