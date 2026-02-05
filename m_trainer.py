# -*- coding: utf-8 -*-

from m_util import conf_parser, model_object_parser, get_model_type, get_path, load_part
from consts import VAE
from m_train_vae import train as train_vae
from m_train_vae_progressive import train_progressive
from torch.utils.tensorboard import SummaryWriter

from torch import optim, nn
import torch
from scheduler import CycleScheduler


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(lr, epoch, sched, optimizer, loader):
    scheduler = None
    if sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, lr, n_iter=len(loader) * epoch, momentum=None
        )
    return scheduler


def train(folder_name, loader, dataset_name, n_run, sample_period, sampler, start_epoch=-1,
          end_epoch=-1, batch_size=-1, sched=None, device='cuda', size=256, lr=-1, amp=None,
          use_progressive=False, level_1_weight=1.0, level_2_weight=1.0, level_3_weight=1.0,
          patience=-1, kl_anneal_epochs=100, max_kl_weight=0.05):

    model_type = get_model_type(folder_name)
    _, train_params = conf_parser(dataset_name, n_run, folder_name)
    model = model_object_parser(dataset_name, n_run, folder_name)
    model = model.to(device)
    args = {}
    if start_epoch > 1:
        try:
            ckpt = load_part(model, start_epoch - 1, device)
        except RuntimeError as e:
            print('not find checkpoint {}'.format(start_epoch - 1))
            return 0

        args = ckpt['args']
        model = ckpt['model']
        if 'lr' in args:
            lr = args['lr']
        if 'batch' in args:
            batch_size = args['batch']
        if 'amp' in args:
            amp = args['amp']

    if lr < 0:
        lr = train_params['lr']
    if start_epoch < 0:
        start_epoch = 0
    if end_epoch < 0:
        end_epoch = train_params['epoch']
    if batch_size < 0:
        batch_size = train_params['batch']
    if amp is None and 'amp' in train_params:
        amp = train_params['amp']

    args['lr'] = lr
    args['batch'] = batch_size
    args['amp'] = amp

    optimizer = get_optimizer(model, lr)

    model = nn.DataParallel(model)
    model = model.to(device)
    sample_iter = 0
    folder_path = get_path(dataset_name, n_run, folder_name, 'ckpt', checkpoint=0)[:-6]
    writer = SummaryWriter(log_dir=folder_path+'{}_{}'.format(*[start_epoch, end_epoch]))

    if model_type == VAE:
        scheduler = get_scheduler(lr, end_epoch - start_epoch, sched, optimizer, loader)

        # Add cosine annealing learning rate scheduler for better convergence
        from torch.optim.lr_scheduler import CosineAnnealingLR
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=end_epoch-start_epoch, eta_min=1e-5)

        # Check if model has SMPLX capability
        use_smplx_loss = False
        if hasattr(model, 'module'):  # DataParallel wrapper
            use_smplx_loss = hasattr(model.module, 'use_smplx') and model.module.use_smplx
        elif hasattr(model, 'use_smplx'):
            use_smplx_loss = model.use_smplx

        # Early stopping setup (only starts after KL annealing is done)
        best_loss = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        early_stop_enabled = patience > 0

        if early_stop_enabled:
            print(f"Early stopping enabled with patience={patience} (starts after KL annealing epoch {kl_anneal_epochs})")
        else:
            print("Early stopping disabled")

        for i in range(start_epoch, end_epoch):
            sample_iter += 1
            do_sample = sample_period > 0 and sample_iter % sample_period ==0

            if use_progressive:
                # Progressive training with level-specific losses
                epoch_loss = train_progressive(folder_name, i, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, n_run,
                                use_smplx_loss=use_smplx_loss,
                                pose_loss_weight=1.0,
                                vertex_loss_weight=5.0,
                                joint_loss_weight=3.0,
                                level_1_weight=level_1_weight,
                                level_2_weight=level_2_weight,
                                level_3_weight=level_3_weight,
                                kl_anneal_epochs=kl_anneal_epochs,
                                max_kl_weight=max_kl_weight)
            else:
                # Standard training with final output only
                epoch_loss = train_vae(folder_name, i, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, n_run,
                           use_smplx_loss=use_smplx_loss,
                           pose_loss_weight=1.0,
                           vertex_loss_weight=5.0,
                           joint_loss_weight=3.0,
                           kl_anneal_epochs=kl_anneal_epochs,
                           max_kl_weight=max_kl_weight)

            # Early stopping check (only after KL annealing is complete)
            kl_annealing_done = (i + 1) >= kl_anneal_epochs
            if kl_annealing_done:
                is_best = epoch_loss < best_loss
                if is_best:
                    best_loss = epoch_loss
                    epochs_without_improvement = 0
                    if early_stop_enabled:
                        best_model_state = model.state_dict()
                        print(f"  New best loss: {best_loss:.6f}")
                else:
                    epochs_without_improvement += 1
            else:
                is_best = True  # always save as "best" during annealing

            # Save checkpoint
            save_path = get_path(dataset_name, n_run, folder_name, 'ckpt', checkpoint=i)
            torch.save(model.state_dict(), save_path)

            # Save best checkpoint
            if is_best:
                best_path = get_path(dataset_name, n_run, folder_name, 'ckpt', checkpoint='best')
                torch.save(model.state_dict(), best_path)

            # Step the cosine annealing scheduler
            if cosine_scheduler is not None:
                cosine_scheduler.step()

            # Check if should stop early (only possible after KL annealing)
            if early_stop_enabled and kl_annealing_done and epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after epoch {i+1}")
                print(f"   No improvement for {patience} epochs (counting started at epoch {kl_anneal_epochs})")
                print(f"   Best loss: {best_loss:.6f} at epoch {i+1-epochs_without_improvement}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    # Save final best model
                    final_path = get_path(dataset_name, n_run, folder_name, 'ckpt', checkpoint=i)
                    torch.save(model.state_dict(), final_path)
                    print(f"   Restored and saved best model to {final_path}")
                break

        writer.close()
