import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import LabelTransformer, StereopsisDataset, np_to_tensor


def create_dataloaders(data_path, batch_size=32, test_split_ratio=0.9):
    label_transformer = LabelTransformer(h=120, w=214)
    dataset = StereopsisDataset(data_path, transform=transforms.Compose([np_to_tensor]),
                                target_transform=transforms.Compose([label_transformer]))

    train_size = int(test_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader, test_dataset.indices


def train(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    seen_samples = 0
    for batch, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction error
        pred = model(x1, x2)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seen_samples += len(x1)
        print(f"\rloss: {loss.item():>7f}  [{seen_samples:>5d}/{size:>5d}]", end="")
    print("\n")


def test(dataloader, model, loss_fn, device="cuda"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred = model(x1, x2)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
