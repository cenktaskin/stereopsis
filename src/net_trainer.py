from pathlib import Path
from tqdm import tqdm
from importlib import import_module

import torch
from torch.utils.data import DataLoader

from pretrained_weights import fetch_pretrained_dispnet_weights


def trainer(model_name, train_dataset, validation_dataset, loss_fn, accuracy_fn, current_device, writer, epochs,
            batch_size, batch_norm, not_pretrained, learning_rate, scheduler_step, scheduler_gamma, num_workers,
            freeze_encoder):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_batch_count = len(train_dataloader)
    val_batch_count = len(val_dataloader)

    results_path = Path(writer.log_dir)
    torch.save(val_dataloader, results_path.joinpath("val_loader.pt"))

    model_net = getattr(import_module(f"models.{model_name}"), "NNModel")
    model = model_net(batch_norm)
    writer.add_graph(model, torch.randn((1, 6, 384, 768), requires_grad=False))
    if not not_pretrained:
        model.load_state_dict(fetch_pretrained_dispnet_weights(model))
        if freeze_encoder:
            model.encoder.requires_grad_(False)
    model = model.to(current_device)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    each_round = epochs // 4
    for i in range(epochs):
        with tqdm(total=train_batch_count, unit="batch", leave=False) as pbar:
            pbar.set_description(f"Epoch [{i:4d}/{epochs:4d}]")

            # Training
            model.train()
            running_train_loss, running_train_epe = 0, 0
            for j, (x, y) in enumerate(train_dataloader):
                batch_idx = i * len(train_dataloader) + j + 1

                x, y = x.to(current_device).float(), y.to(current_device).float()
                predictions = model(x)
                loss = loss_fn(predictions, y, i // each_round)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss = loss.item()
                running_train_loss += train_loss
                running_train_epe += accuracy_fn(predictions, y).item()

                writer.add_scalar("RunningLoss/iteration", train_loss, batch_idx)
                pbar.set_postfix(loss=f"{train_loss:.4f}")
                pbar.update(True)

            scheduler.step()
            model.eval()
            running_val_epe, running_val_loss = 0, 0
            with torch.no_grad():
                for k, (x, y) in enumerate(val_dataloader):
                    x, y = x.to(current_device).float(), y.to(current_device).float()
                    predictions = model(x)
                    running_val_loss += loss_fn(predictions, y, stage=3).item()
                    running_val_epe += accuracy_fn(predictions, y).item()

            avg_train_epe = running_train_epe / train_batch_count
            avg_val_epe = running_val_epe / val_batch_count

            avg_train_loss = running_train_loss / train_batch_count
            avg_val_loss = running_val_loss / val_batch_count

            writer.add_scalars('Loss/epoch', {'Training': avg_train_loss, 'Validation': avg_val_loss}, i)
            writer.add_scalars('Error/epoch', {'Training': avg_train_epe, 'Validation': avg_val_epe}, i)
            writer.flush()

        if i % each_round == each_round - 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

        if i == (epochs // 2):
            torch.save(model.state_dict(), results_path.joinpath(f"model-e{i}.pt"))

    torch.save(model.state_dict(), results_path.joinpath(f"model-e{epochs}.pt"))
