from torch import nn
from tqdm import tqdm


def train(folder_name, epoch_num, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, run_num):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25

    mse_sum = 0
    mse_n = 0

    for i, (data, label) in enumerate(loader):
        model.zero_grad()

        data = data.to(device)

        out, latent_loss = model(data)
        recon_loss = criterion(out, data)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * data.shape[0]
        mse_n += data.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch_num + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

    writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)

    if do_sample:
        sampler(folder_name, model, data, dataset_name, run_num, epoch_num, data.shape[0])
