# Final Approach: HR-VQVAE-POSE

## Overview

Three-stage pipeline: (1) VAE (pose tokenizer), (2) Allspk flow matching prior pretraining,
(3) Speaker-specific E2E perceptual fine-tuning.

---

## Stage 1: VAE (Pose Tokenizer)

**Script:** `m_train.py` (calls `m_train_vae_progressive.py` + `m_trainer.py`)
**Model:** `m_vae_pose.py` — `LengthConditionedVAE` with AdaLN decoder + temporal downsampling
**Config:** `checkpoint/beat2_poses/0/vae_temporal_lean/conf.ini`

### Architecture Parameters
| Parameter | Value |
|---|---|
| `in_channel` | 165 (55 joints × 3 axis-angle) |
| `d_model` | 512 |
| `latent_dim` | 16 |
| `embed_dim` | 16 |
| `n_level` | 3 (progressive: body → face → hands) |
| `nhead` | 8 |
| `num_encoder_layers` | 6 |
| `num_decoder_layers` | 6 |
| `dim_feedforward` | 2048 |
| `dropout` | 0.1 |
| `temporal_downsample` | 8× |
| `num_memory_tokens` | 4 |
| `anchor_max_frames` | 90 |
| `anchor_prob` | 0.5 |
| `num_speakers` | 31 |

### Training Parameters
| Parameter | Value |
|---|---|
| `batch` | 8 |
| `epoch` | 400 |
| `lr` | 1e-4 |
| Checkpoint selection | min val reconstruction loss |

### VAE Checkpoints
- **Allspk VAE:** `checkpoint/beat2_poses/0/vae_temporal_lean/best.pt`
- **Spk2 fine-tuned VAE:** `checkpoint/beat2_poses/0/vae_lean_dec_spk2/best.pt`
  - Spk2 FT script: `finetune_vae_decoder_spk2.py`
  - Checkpoint selected by val reconstruction loss (epoch 24)

---

## Stage 2: Allspk Flow Matching Prior Pretraining

**Script:** `prior_net/train_diffusion_online.py`
**Model:** `prior_net/diffusion_model.py` — `TemporalFlowMatchingPrior`

### Architecture Parameters
| Parameter | Value |
|---|---|
| `d_model` | 192 |
| `nhead` | 4 |
| `num_audio_layers` | 3 |
| `num_denoiser_layers` | 4 |
| `dim_feedforward` | 768 |
| Text conditioning | enabled (BERT features) |
| Temporal mode | enabled (`--temporal`) |

### Training Parameters
| Parameter | Value |
|---|---|
| `--checkpoint-dir` | `e2e_perceptual/checkpoints_moment_allspk_clean` |
| `--vae-checkpoint` | `checkpoint/beat2_poses/0/vae_temporal_lean/best.pt` |
| `--folder-name` | `vae_temporal_lean` |
| `--temporal` | enabled |
| `--val-autoreg-every` | 5 |
| `--val-autoreg-fraction` | 1.0 |
| Checkpoint selection | min val AR FGD on **val split** (`split='val'`) |

### Result
- Best checkpoint: `e2e_perceptual/checkpoints_moment_allspk_clean/best.pt` (epoch 59)
- Val AR FGD (spk2, g=1.0, val split): **0.7765**

---

## Stage 3: E2E Perceptual Fine-tuning (Allspk → Spk2)

**Script:** `e2e_perceptual/train.py`
**Model:** same `TemporalFlowMatchingPrior` as Stage 2
**Warm start:** `--resume e2e_perceptual/checkpoints_moment_allspk_clean/best.pt`

### Training Parameters
| Parameter | Value |
|---|---|
| `--checkpoint-dir` | `e2e_perceptual/checkpoints_moment_spk2_clean` |
| `--vae-checkpoint` | `checkpoint/beat2_poses/0/vae_lean_dec_spk2/best.pt` |
| `--folder-name` | `vae_temporal_lean` |
| `--speaker` | 2 |
| `--temporal` | enabled |
| `--lr` | 1e-5 |
| `--lambda-fgd` | 0.01 |
| `--lambda-beat` | 0.1 |
| `--lambda-cov` | 1.0 |
| `--val-autoreg-every` | 1 |
| ODE steps (perceptual) | 5 (training), 50 (val AR FGD) |
| Checkpoint selection | min val AR FGD on **val split** (`split='val'`, spk2 only) |

### Combined Loss
```
L = flow_matching_loss
  + 0.01 * fgd_feature_matching_loss
  + 0.1  * beat_alignment_loss
  + 1.0  * covariance_loss
```

### Result
- Best checkpoint: `e2e_perceptual/checkpoints_moment_spk2_clean/best.pt` (epoch 1)
- Val AR FGD (spk2, g=1.0, val split): **0.7097**

---

## Stage 4: Guidance Scale Selection (Val Set)

**Script:** `prior_net/evaluate_diffusion.py`
**Split:** `--split val` (never touches test set)

```bash
for g in 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7; do
  python prior_net/evaluate_diffusion.py \
    --prior-checkpoint e2e_perceptual/checkpoints_moment_spk2_clean/best.pt \
    --vae-checkpoint checkpoint/beat2_poses/0/vae_lean_dec_spk2/best.pt \
    --folder-name vae_temporal_lean --temporal --speaker 2 \
    --test-fraction 1.0 --split val --guidance-scale $g \
    --num-steps 50 --seed 42 \
    --output e2e_perceptual/checkpoints_moment_spk2_clean/val_sweep_g${g}.json
done
```

**Best guidance scale: g = 0.5** (val FGD = 0.5979)

---

## Stage 5: Final Test Evaluation

**Script:** `prior_net/evaluate_diffusion.py`
**Split:** `--split test`
**Seeds:** 42, 123, 456, 789, 1337
**Steps:** 50

### Spk2 Results (speaker 2 test set, 15 recordings)
| Metric | Mean ± Std |
|---|---|
| FGD | **0.4374 ± 0.0067** |
| BC | 0.6944 ± 0.0060 |
| L1Div | 14.85 ± 0.25 |

### All-Speaker Results (full test set, 265 recordings)
| Metric | Mean ± Std |
|---|---|
| FGD | **0.3416 ± 0.0073** |
| BC | 0.4348 ± 0.0016 |
| L1Div | 9.311 ± 0.049 |

---

## Reconstruction Upper Bound (VAE ceiling)

**Script:** `evaluate_reconstruction.py`
Uses encoder mu (deterministic) instead of sampled z. Autoregressive with decoded anchors.

### Spk2 VAE (speaker 2 test set, 15 recordings)
| Metric | Value |
|---|---|
| FGD | 0.269 |
| BC | 0.664 |
| MPJPE (all) | 32.66 mm |
| MPJPE body | 21.29 mm |
| MPJPE face | 17.54 mm |
| MPJPE left hand | 41.71 mm |
| MPJPE right hand | 43.32 mm |

### Allspk VAE (full test set, 265 recordings)
| Metric | Value |
|---|---|
| FGD | 0.194 |
| BC | 0.400 |
| MPJPE (all) | 26.03 mm |
| MPJPE body | 17.71 mm |
| MPJPE face | 13.32 mm |
| MPJPE left hand | 33.36 mm |
| MPJPE right hand | 33.46 mm |

---

## Ablation Study (spk2 test set, 5 seeds, g=0.5)

### What was tested
Two component ablations on the full model (E2E perceptual spk2 FT, `vae_lean_dec_spk2`):

| Ablation | How it was run |
|---|---|
| **w/o anchor prefix** | `prior_net/evaluate_diffusion.py --anchor-frames 0` — sets `anchor_k=0` in `generate_autoregressive_with_diffusion`, disabling anchor frames for both the VAE decoder and the prior's conditioning |
| **w/o flow-matching prior** | `evaluate_generation.py` directly — uses `generate_autoregressive` which samples z ~ N(0, I) without any diffusion prior |

### Results

| Model | FGD ↓ | BC ↑ | L1Div ↑ |
|---|---|---|---|
| **Full model** | **0.4374 ± 0.0067** | 0.6944 ± 0.0060 | 14.85 ± 0.25 |
| w/o anchor prefix | 0.5183 ± 0.0113 | 0.6721 ± 0.0056 | 14.77 ± 0.13 |
| w/o flow-matching prior (z ~ N(0,I)) | 0.7384 ± 0.0072 | 0.7681 ± 0.0025 | 9.94 ± 0.06 |

### Observations
- **Anchor prefix** contributes +0.08 FGD (18% degradation) — important for temporal coherence across autoregressive chunks
- **Flow-matching prior** contributes +0.30 FGD (69% degradation) — the dominant component; without it the model outputs incoherent random-z motions
- BC for no-prior is *higher* than the full model — random z produces over-animated motions that spuriously correlate with audio beats, but the overall gesture distribution (FGD) is far off

---

## Key Files

| File | Purpose |
|---|---|
| `m_vae_pose.py` | VAE model (LengthConditionedVAE, AdaLN decoder) |
| `m_train.py` | VAE training entry point |
| `m_train_vae_progressive.py` | Progressive 3-level VAE training logic |
| `m_trainer.py` | Training loop, checkpoint selection by val loss |
| `m_beat_dataset.py` | BEAT2 dataset (train/val/test split from CSV) |
| `m_conf_parser.py` | Config parsing from conf.ini |
| `m_util.py` | Model loading utilities |
| `prior_net/diffusion_model.py` | TemporalFlowMatchingPrior model |
| `prior_net/train_diffusion_online.py` | Allspk prior pretraining |
| `prior_net/evaluate_diffusion.py` | Generation evaluation (FGD/BC/L1Div) |
| `e2e_perceptual/train.py` | E2E perceptual FT (FM + FGD + beat + cov losses) |
| `evaluate_generation.py` | `get_test_files()` with val/test split support |
| `evaluate_reconstruction.py` | Reconstruction eval with per-level MPJPE |
| `generate_autoregressive_v2.py` | Chunked autoregressive generation with anchors |
| `finetune_vae_decoder_spk2.py` | VAE decoder spk2 fine-tuning |
| `checkpoint/beat2_poses/0/vae_temporal_lean/conf.ini` | VAE architecture config |

---

## Leakage-Free Protocol

All checkpoint selection uses the **val split** exclusively:
- VAE `best.pt`: selected by val reconstruction loss (`split='val'` in `BEAT2PoseDataset`)
- Prior `best_ar.pt`: selected by val AR FGD (`get_test_files(..., split='val')`)
- E2E FT `best.pt`: selected by val AR FGD only (batch FGD never selects checkpoints)
- Guidance scale: swept on val set (`--split val`), applied to test set
- Final metrics: test set only, 5 seeds for mean ± std
