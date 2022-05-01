import torch

sh_check_repo = 'if [ -d stereopsis ]; then echo "Repo already cloned, pulling"; cd stereopsis; git checkout ' \
                '$current_branch ; git pull ;  else git clone -b $current_branch $URL ; fi '


def train(dataloader, model, loss_fn, optimizer, device="cuda"):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        # Compute prediction error
        pred = model(x1, x2)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 4 == 0:
            loss, current = loss.item(), batch * len(x1)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
