# -*- coding: utf-8 -*-

import csv
import os

from m_util import conf_parser, model_object_parser, get_model_type, get_path, load_part
from consts import VAE
from m_train_vae import train as train_vae
from m_train_vae_progressive import train_progressive
from torch.utils.tensorboard import SummaryWriter

from torch import optim, nn
import torch
from scheduler import CycleScheduler


def validate_epoch(model, val_loader, device):
    """
    Compute masked pose MSE on validation set (no KL, no geometry losses).
    Returns average validation reconstruction loss.
    """
    model.eval()
    val_loss_sum = 0.0
    val_n = 0

    with torch.no_grad():
        for data, label in val_loader:
            if isinstance(data, dict):
                poses = data['poses'].to(device)
                gesture_type = data.get('gesture_type', None)
                if gesture_type is not None:
                    gesture_type = gesture_type.to(device)
                lengths = data.get('lengths', None)
                if lengths is not None:
                    lengths = lengths.to(device)
                padding_mask = data.get('padding_mask', None)
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)
                audio = data.get('audio', None)
                if audio is not None:
                    audio = audio.to(device)
                speaker_id = data.get('speaker_id', None)
                if speaker_id is not None:
                    speaker_id = speaker_id.to(device)
                anchor_pool = data.get('anchor_pool', None)
                if anchor_pool is not None:
                    anchor_pool = anchor_pool.to(device)
                anchor_audio = data.get('anchor_audio', None)
                if anchor_audio is not None:
                    anchor_audio = anchor_audio.to(device)
            else:
                poses = data.to(device)
                padding_mask = None
                gesture_type = None
                lengths = None
                audio = None
                speaker_id = None
                anchor_pool = None
                anchor_audio = None

            result = model(poses, padding_mask=padding_mask, gesture_type=gesture_type,
                          lengths=lengths, audio_features=audio, speaker_id=speaker_id,
                          anchor_pool=anchor_pool, anchor_audio=anchor_audio)
            # Handle both (recon, kl) and (intermediates, kl) return formats
            if isinstance(result[0], list):
                reconstructed_poses = result[0][-1]  # final level output
            else:
                reconstructed_poses = result[0]

            # Masked pose MSE only
            if padding_mask is not None:
                valid_mask = ~padding_mask
                valid_mask_expanded = valid_mask.unsqueeze(-1)
                num_valid = valid_mask.sum().clamp(min=1)
                batch_loss = (((reconstructed_poses - poses) ** 2) * valid_mask_expanded).sum() / (num_valid * poses.shape[-1])
            else:
                batch_loss = torch.nn.functional.mse_loss(reconstructed_poses, poses)

            batch_size = poses.shape[0]
            val_loss_sum += batch_loss.item() * batch_size
            val_n += batch_size

    model.train()
    return val_loss_sum / val_n if val_n > 0 else float('inf')


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
          patience=-1, kl_anneal_epochs=100, max_kl_weight=0.05, val_loader=None,
          vel_weight=1.0, cont_vel_weight=50.0,
          anchor_recon_weight=0.1):

    model_type = get_model_type(folder_name)
    _, train_params = conf_parser(dataset_name, n_run, folder_name)
    model = model_object_parser(dataset_name, n_run, folder_name)
    model = model.to(device)
    args = {}
    if start_epoch > 0:
        ckpt_path = get_path(dataset_name, n_run, folder_name, 'ckpt', checkpoint=start_epoch - 1)
        if os.path.exists(ckpt_path):
            print(f"Loading checkpoint: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location=device)
            # Strip 'module.' prefix if present (saved from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k[len('module.'):] if k.startswith('module.') else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        else:
            print(f"Checkpoint not found: {ckpt_path}")
            return 0

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

        # Warmup + cosine annealing LR schedule (always based on total epochs)
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_epochs = 20
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, end_epoch - warmup_epochs), eta_min=1e-5)
        combined_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        # Fast-forward LR scheduler if resuming
        if start_epoch > 0:
            for _ in range(start_epoch):
                combined_scheduler.step()
            print(f"LR scheduler fast-forwarded to epoch {start_epoch} (lr={optimizer.param_groups[0]['lr']:.6f})")

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
            stop_metric = "val_recon" if val_loader is not None else "train_total"
            print(f"Early stopping enabled with patience={patience} on {stop_metric} (starts after KL annealing epoch {kl_anneal_epochs})")
        else:
            print("Early stopping disabled")

        if val_loader is not None:
            print(f"Validation loader provided ({len(val_loader)} batches)")

        # ===== CSV loss log setup =====
        loss_log_path = os.path.join(folder_path, 'loss_log.csv')
        loss_log_header_written = False
        loss_log_keys = None

        # When resuming, read existing CSV to restore early stopping state and append mode
        if start_epoch > 0 and os.path.exists(loss_log_path):
            with open(loss_log_path) as f:
                existing_rows = list(csv.DictReader(f))
            if existing_rows:
                loss_log_keys = list(existing_rows[0].keys())
                loss_log_header_written = True  # will append, not overwrite
                # Restore best_loss from existing log for early stopping
                for row in existing_rows:
                    val = row.get('val_loss', '')
                    if val and val != '':
                        loss_val = float(val)
                    else:
                        loss_val = float(row.get('total', float('inf')))
                    if loss_val < best_loss:
                        best_loss = loss_val
                print(f"Restored best_loss={best_loss:.6f} from existing log ({len(existing_rows)} epochs)")

        for i in range(start_epoch, end_epoch):
            sample_iter += 1
            do_sample = sample_period > 0 and sample_iter % sample_period ==0

            if use_progressive:
                # Progressive training with level-specific losses
                epoch_metrics = train_progressive(folder_name, i, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, n_run,
                                use_smplx_loss=use_smplx_loss,
                                pose_loss_weight=1.0,
                                vertex_loss_weight=5.0,
                                joint_loss_weight=3.0,
                                level_1_weight=level_1_weight,
                                level_2_weight=level_2_weight,
                                level_3_weight=level_3_weight,
                                kl_anneal_epochs=kl_anneal_epochs,
                                max_kl_weight=max_kl_weight,
                                vel_weight=vel_weight,
                                cont_vel_weight=cont_vel_weight,
                                anchor_recon_weight=anchor_recon_weight)
                epoch_loss = epoch_metrics['total']
            else:
                # Standard training with final output only
                epoch_loss = train_vae(folder_name, i, loader, model, writer, do_sample, sampler, optimizer, scheduler, device, dataset_name, n_run,
                           use_smplx_loss=use_smplx_loss,
                           pose_loss_weight=1.0,
                           vertex_loss_weight=5.0,
                           joint_loss_weight=3.0,
                           kl_anneal_epochs=kl_anneal_epochs,
                           max_kl_weight=max_kl_weight)
                epoch_metrics = {'total': epoch_loss}

            # Early stopping check (only after KL annealing is complete)
            kl_annealing_done = (i + 1) >= kl_anneal_epochs

            # Compute val loss if val_loader is provided and KL annealing is done
            val_loss = None
            if val_loader is not None and kl_annealing_done:
                val_loss = validate_epoch(model, val_loader, device)
                writer.add_scalar('Loss/val_recon', val_loss, i)
                print(f"  Epoch {i+1}: train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

            # ===== Write CSV loss log =====
            epoch_metrics['epoch'] = i + 1
            epoch_metrics['val_loss'] = val_loss if val_loss is not None else ''
            if not loss_log_header_written:
                # First epoch: overwrite file with header
                # Put epoch first, then sorted keys for consistent column order
                all_keys = ['epoch'] + sorted(k for k in epoch_metrics if k != 'epoch')
                with open(loss_log_path, 'w', newline='') as f:
                    writer_csv = csv.DictWriter(f, fieldnames=all_keys)
                    writer_csv.writeheader()
                    writer_csv.writerow(epoch_metrics)
                loss_log_header_written = True
                loss_log_keys = all_keys
            else:
                with open(loss_log_path, 'a', newline='') as f:
                    writer_csv = csv.DictWriter(f, fieldnames=loss_log_keys)
                    writer_csv.writerow(epoch_metrics)

            if kl_annealing_done:
                # Use val loss for early stopping if available, otherwise train loss
                stop_loss = val_loss if val_loss is not None else epoch_loss
                is_best = stop_loss < best_loss
                if is_best:
                    best_loss = stop_loss
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

            # Step the LR scheduler (warmup → cosine)
            combined_scheduler.step()

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
