from tqdm import tqdm
from importlib import import_module

import torch
from torch.utils.data import DataLoader

from loss import MultilayerSmoothL1, MaskedEPE


def model_trainer(model_name, train_dataset, validation_dataset, current_device, epochs, batch_size, batch_norm,
                  pretrained, learning_rate, scheduler_step, scheduler_gamma, writer):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_batch_count = len(train_dataloader)
    val_batch_count = len(val_dataloader)

    model_net = getattr(import_module(f"models.{model_name}"), "NNModel")
    model = model_net(batch_norm)
    writer.add_graph(model, torch.randn((1, 6, 384, 768), requires_grad=False))
    if pretrained:
        model.ingest_pretrained_weights()
    model = model.to(current_device)

    loss_fn = MultilayerSmoothL1()
    accuracy_fn = MaskedEPE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # was 0.05 on original paper but it is exploding
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # each_round = epochs // 4
    for i in range(epochs):
        with tqdm(total=train_batch_count, unit="batch", leave=False) as pbar:
            pbar.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

            # Training
            model.train()
            running_train_loss, running_train_epe = 0, 0
            for j, (x, y) in enumerate(train_dataloader):
                batch_idx = i * len(train_dataloader) + j + 1

                x, y = x.to(current_device), y.to(current_device)
                predictions = model(x)
                loss = loss_fn(predictions, y, batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
                running_train_loss += train_loss
                running_train_epe += accuracy_fn(predictions, y).item()

                writer.add_scalar("Running loss/iteration", train_loss, batch_idx)
                pbar.set_postfix(loss=f"{train_loss:.4f}")
                pbar.update(True)

            scheduler.step()

            model.eval()
            running_val_epe, running_val_loss = 0, 0
            with torch.no_grad():
                for k, (x, y) in enumerate(val_dataloader):
                    x, y = x.to(current_device), y.to(current_device)
                    predictions = model(x)
                    running_val_loss += loss_fn(predictions, y, 10 ** 10).item()
                    running_val_epe += accuracy_fn(predictions, y).item()

            avg_train_epe = running_train_epe / train_batch_count
            avg_val_epe = running_val_epe / val_batch_count

            avg_train_loss = running_train_loss / train_batch_count
            avg_val_loss = running_val_loss / val_batch_count

            writer.add_scalars('Loss/epoch', {'Training': avg_train_loss, 'Validation': avg_val_loss}, i)
            writer.add_scalars('Accuracy/epoch', {'Training': avg_train_epe, 'Validation': avg_val_epe}, i)
            writer.flush()

        # if i % each_round == each_round - 1:
        #    optimizer = torch.optim.Adam(model.parameters(), lr=10 ** -4)
        #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # save every round

    return model, val_dataloader
