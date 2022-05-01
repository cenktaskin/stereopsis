# Show a sample form dataset
train_features_l, train_features_r, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features_l.size()}")
print(f"Labels batch shape: {train_labels.size()}")
x_l = train_features_l[0].squeeze()
x_r = train_features_r[0].squeeze()
y = train_labels[0]
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.title(f"Training sample with type {type(x_l)}, shape {x_l.shape}")
plt.imshow(x_l.permute(1, 2, 0))  # back to np dimension order
plt.subplot(2, 2, 2)
#plt.title(f"Training sample with type {type(x_l)}, shape {x_l.shape}")
plt.imshow(x_r.permute(1, 2, 0))  # back to np dimension order
plt.subplot(2, 2, 3)
plt.title(f"Training sample with type {type(y)}, shape {y.shape}")
plt.imshow(y)
plt.show()


# model summary
import torchsummary
X = torch.rand(1, 3, 720, 1280, device=device)
logits = model(X,X)
torchsummary.summary(model, [(3,720,1280),(3,720,1280)])


# some tests
for i in range(5):
    current_img_count = len(list(output_dir.glob('test*')))
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
            fig.savefig(output_dir.joinpath(f"test{current_img_count}.jpg"))
            current_img_count += 1

