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

### Allspk VAE on spk2 test (speaker 2 test set, 15 recordings)
| Metric | Value |
|---|---|
| FGD | 0.3232 |
| BC | 0.6571 |
| MPJPE (all) | 34.84 mm |
| MPJPE body | 22.76 mm |
| MPJPE face | 18.45 mm |
| MPJPE left hand | 44.6 mm |
| MPJPE right hand | 46.1 mm |

Note: spk2-finetuned VAE gives MPJPE 32.66mm vs 34.84mm here — only ~2mm gain from speaker FT.

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
Three component ablations on the full model (E2E perceptual spk2 FT, `vae_lean_dec_spk2`):

| Ablation | How it was run |
|---|---|
| **w/o anchor prefix** | `prior_net/evaluate_diffusion.py --anchor-frames 0` — sets `anchor_k=0` in `generate_autoregressive_with_diffusion`, disabling anchor frames for both the VAE decoder and the prior's conditioning |
| **w/o flow-matching prior** | `evaluate_generation.py` directly — uses `generate_autoregressive` which samples z ~ N(0, I) without any diffusion prior |
| **all-beat labels** | `prior_net/evaluate_diffusion.py --all-beat` — overrides all chunk `gesture_type` to 0 (beat) regardless of GT `.sem` file; baseline for gesture-type conditioning |
| **w/o E2E perceptual FT** | `prior_net/evaluate_diffusion.py` with `--prior-checkpoint e2e_perceptual/checkpoints_moment_allspk_clean/best.pt --vae-checkpoint checkpoint/beat2_poses/0/vae_temporal_lean/best.pt` — uses allspk prior + allspk VAE directly without spk2 E2E FT |

### Results

| Model | FGD ↓ | BC ↑ | L1Div ↑ |
|---|---|---|---|
| **Full model** | **0.4374 ± 0.0067** | 0.6944 ± 0.0060 | 14.85 ± 0.25 |
| w/o anchor prefix | 0.5183 ± 0.0113 | 0.6721 ± 0.0056 | 14.77 ± 0.13 |
| w/o flow-matching prior (z ~ N(0,I)) | 0.7384 ± 0.0072 | 0.7681 ± 0.0025 | 9.94 ± 0.06 |
| all-beat labels (no gesture-type info) | 0.4337 ± 0.0072 | 0.6926 ± 0.0054 | 14.54 ± 0.25 |
| all-semantic labels | 0.5882 ± 0.0213 | 0.7019 ± 0.0062 | 15.99 ± 0.38 |
| w/o E2E perceptual FT (allspk prior) | 0.5373 ± 0.0089 | 0.6791 ± 0.0039 | 15.10 ± 0.21 |

### Continuation Loss Ablation (VAE boundary smoothness)

Evaluated on speaker 2 test set (15 recordings) using autoregressive reconstruction (encode GT → decode with decoded anchors). Script: `eval_boundary_metrics.py`.

| Metric | w/o cont. loss | w/ cont. loss |
|---|---|---|
| Boundary velocity error ↓ (mm) | 33.37 | **16.64** |
| Boundary jerk ↓ (mm) | 37.88 | **17.31** |
| Within-chunk MPJPE ↓ (mm) | 34.71 | **29.63** |

- VAE: `vae_temporal_lean_no_cont/best.pt` (ep84, trained with `--cont-vel-weight 0.0 --anchor-recon-weight 0.0`) vs `vae_temporal_lean/best.pt`
- **Boundary velocity/jerk error doubles** without the continuation loss — chunk transitions become visually jarring
- Within-chunk MPJPE also degrades (34.71 vs 29.63mm), suggesting the continuation loss regularizes overall reconstruction quality beyond just boundaries

### Observations
- **Anchor prefix** contributes +0.08 FGD (18% degradation) — important for temporal coherence across autoregressive chunks
- **Flow-matching prior** contributes +0.30 FGD (69% degradation) — the dominant component; without it the model outputs incoherent random-z motions
- BC for no-prior is *higher* than the full model — random z produces over-animated motions that spuriously correlate with audio beats, but the overall gesture distribution (FGD) is far off
- **E2E perceptual FT** contributes +0.10 FGD improvement (19% degradation without it, 0.5373 → 0.4374) — the spk2-specific FT meaningfully closes the gap between allspk and spk2 distribution

---

## Comparative Analysis: HR-VQVAE vs. EMAGE (Ours vs. Baseline)

To validate the structural advantage of the HR-VQVAE architecture, a direct apples-to-apples evaluation was performed against the official pre-trained EMAGE models (`H-Liu1997/emage_audio` from Hugging Face). The evaluation utilized the exact same chunked autoregressive evaluation script (`evaluate_reconstruction.py`), ensuring that both models were supplied with the exact same ground-truth contextual data (expressions, root translation, and foot contacts) during encoding.

### Reconstruction Upper Bound (VQVAE Tokenizer)

*Note: The official EMAGE VQVAE weights appear to have been optimized solely for Speaker 2 (indicated by a `speaker_dims: 1` configuration), which explains the severe performance degradation on the All-Speaker dataset.*

**Speaker 2 Test Set (15 recordings)**
| Metric | HR-VQVAE (Spk2 FT) | EMAGE VQVAE (Pre-trained) | SynTalker RVQVAE (Official) |
|---|---|---|---|
| FGD ↓ | **0.269** | 0.4394 | 0.1660 |
| BC ↑ | 0.664 | **0.7853** | 0.6974 |
| MPJPE (all) ↓ | **32.66 mm** | 90.55 mm | 55.72 mm |
| MPJPE body ↓ | **21.29 mm** | 54.74 mm | 24.49 mm |
| MPJPE hands ↓ | **42.52 mm** | 120.19 mm | 79.62 mm |
| MPJPE face ↓ | **17.54 mm** | - | 25.91 mm |

*SynTalker metrics evaluated using their official protocol: full-recording VQ encode/decode, 53 joints (excl. eyes), zero global translation, per-recording averaging. Their self-reported MPJPE is 54.81mm; our reproduction gives 55.72mm. Script: `evaluate_syntalker_final.py`.*

**All-Speaker Test Set (265 recordings)**
| Metric | HR-VQVAE (Allspk) | EMAGE VQVAE (Pre-trained) | SynTalker RVQVAE (Official) |
|---|---|---|---|
| FGD ↓ | **0.194** | 2.1317 | 1.8582* |
| BC ↑ | 0.400 | **0.6267** | 0.8143 |
| MPJPE (all) ↓ | **26.03 mm** | 112.25 mm | 113.66 mm* |
| MPJPE body ↓ | **17.71 mm** | 71.55 mm | 55.57 mm* |
| MPJPE hands ↓ | **33.36 mm** | 145.22 mm | 157.72 mm* |
| MPJPE face ↓ | **13.32 mm** | 35.12 mm | 70.16 mm* |

### Full Generation Pipeline (Audio -> Pose)

Evaluation of the full audio-to-pose generation on the Speaker 2 test set, comparing the official EMAGE audio model to the E2E fine-tuned HR-VQVAE-POSE prior.

**Speaker 2 Test Set (15 recordings)**
| Model | FGD ↓ | BC ↑ | L1Div ↑ |
|---|---|---|---|
| **HR-VQVAE-POSE (Spk2 FT)** | **0.4374** | 0.6944 | 14.85 |
| EMAGE (Allspk, Ours) | 2.4169 | 0.4651 | **16.23** |

**All-Speaker Test Set (265 recordings)**
| Model | FGD ↓ | BC ↑ | L1Div ↑ |
|---|---|---|---|
| **HR-VQVAE-POSE (Allspk)** | **0.3416** | **0.4348** | **9.31** |
| EMAGE (Official Pre-trained) | 2.5122 | 0.2811 | 7.95 |
| EMAGE (Allspk, Ours) | 2.7199 | 0.3039 | 8.42 |

**Conclusions:**
1. **Holistic Consistency:** By utilizing a progressive 3-level learning approach rather than completely isolated VQ components for different body parts, the HR-VQVAE maintains massive improvements in spatial accuracy (MPJPE ~32mm vs ~90mm). 
2. **Generalization:** HR-VQVAE acts as a true, generalized pose tokenizer across the entire BEAT2 dataset.
3. **Perceptual Realism:** The final generated motions from HR-VQVAE are perceptually much closer to the ground truth distribution (FGD 0.34 vs 2.72 on all-speaker test), with higher diversity and temporal coherence.
4. **EMAGE Limitations:** The EMAGE architecture (PantoMatrix) struggles with the diversity of the all-speaker dataset, even when trained for 100 epochs. The isolated body-part codebooks and complex fusion mechanism may hinder convergence on heterogeneous multi-speaker data.

---

## 4x Temporal Downsampling Experiment

Attempt to reproduce the clean run pipeline with `vae_temporal_4x` (temporal_downsample=4) instead of `vae_temporal_lean` (temporal_downsample=8), to test whether the better VAE reconstruction ceiling translates to better generation quality.

### Pipeline (mirrors clean run exactly)

| Stage | Script | Checkpoint | Result |
|---|---|---|---|
| VAE training | `m_train.py` | `vae_temporal_4x/best.pt` | Recon FGD=0.204, MPJPE=28.86mm |
| Decoder FT (spk2) | `finetune_vae_decoder_4x_spk2.py` | `vae_4x_dec_spk2/best.pt` (ep37) | val_recon=0.0020 |
| Allspk prior | `prior_net/train_diffusion_online.py` | `checkpoints_moment_allspk_4x_v2/best_ar.pt` (ep90) | val AR FGD=0.3622 |
| E2E spk2 FT | `e2e_perceptual/train.py` | `checkpoints_moment_spk2_4x_v2/best.pt` (ep5) | val AR FGD=0.7022 |
| Guidance sweep | `prior_net/evaluate_diffusion.py --split val` | best g=0.4 | val FGD=0.5111 |

### Final Test Results (g=0.4, 5 seeds, speaker 2 test set)

| Metric | 4x pipeline | 8x clean run |
|---|---|---|
| FGD ↓ | 0.4613 ± 0.0235 | **0.4374 ± 0.0067** |
| BC ↑ | **0.7155 ± 0.0033** | 0.6944 ± 0.0060 |
| L1Div ↑ | **15.20 ± 0.33** | 14.85 ± 0.25 |

### Key Findings

- The 4x VAE has a substantially better reconstruction ceiling (FGD 0.204 vs 0.323, MPJPE 28.86 vs 34.84mm) but **does not improve final generation FGD** (0.4613 vs 0.4374).
- The bottleneck is the prior/E2E FT quality, not the VAE decoder. With 37-token latent sequences (vs 19 for 8x), the prior has more to learn and convergence is slower.
- BC and L1Div are slightly better for 4x, suggesting more natural motion dynamics, but FGD (distribution-level quality) is worse.
- Optimal guidance scale shifted from g=0.5 (8x) to g=0.4 (4x), with a flat plateau g=0.3–0.5.
- **Conclusion:** 8x temporal downsampling is a better trade-off for this architecture — shorter latent sequences are easier for the flow-matching prior to learn, and the reconstruction quality difference does not compensate.

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

## LLM Gesture Planner Evaluation (Spk2)

Tests whether LLM-predicted gesture-type labels (beat vs. semantic) can replace GT BEAT2 `.sem` labels at inference time.

### Setup

| Component | Value |
|---|---|
| VAE | `checkpoint/beat2_poses/0/vae_lean_dec_spk2/best.pt` |
| Prior | `e2e_perceptual/checkpoints_moment_spk2_clean/best.pt` |
| LLM planner | Qwen3.5-9B (few-shot) or Qwen3.5-9B + LoRA fine-tuned |
| Guidance scale | 0.5 |
| Seeds | 42, 123, 456, 789, 1337 |
| Test set | spk2, 15 recordings |
| Planner script | `batch_plan_gestures.py` (reads BEAT2 train_test_split.csv) |
| Eval script | `prior_net/evaluate_diffusion.py --sem-dir <dir>` |

Predicted `.txt` sem files saved to `planned_sem_fewshot/` and `planned_sem_lora/`.
`--sem-dir` added to `prior_net/evaluate_diffusion.py` and `generate_autoregressive_v2.py::load_recording()` to override GT labels.

### Results

| Label source | FGD ↓ | BC ↑ | L1Div ↑ |
|---|---|---|---|
| GT BEAT2 `.sem` labels | **0.4374 ± 0.0067** | 0.6944 ± 0.0060 | 14.85 ± 0.25 |
| All-beat (no gesture-type info) | 0.4337 ± 0.0072 | 0.6926 ± 0.0054 | 14.54 ± 0.25 |
| LLM few-shot (Qwen3.5-9B) | 0.4575 ± 0.0149 | 0.6906 ± 0.0059 | 14.69 ± 0.16 |
| LLM LoRA fine-tuned | 0.4634 ± 0.0107 | 0.7132 ± 0.0065 | 15.15 ± 0.45 |

### Planner Quality Metrics (vs. GT `.sem` labels, 15 spk2 test recordings)

**Script:** `evaluate_planner.py`

| Metric | Few-shot | LoRA fine-tuned |
|---|---|---|
| Per-frame accuracy ↑ | **75.0%** | 57.8% |
| Macro-F1 (beat + semantic) ↑ | **52.1%** | 46.5% |
|   Beat F1 | **84.4%** | 69.0% |
|   Semantic F1 | 19.8% | **23.9%** |
| Boundary F1 @0.5s ↑ | 19.1% | **35.1%** |
|   Boundary Precision | **38.6%** | 27.6% |
|   Boundary Recall | 15.4% | **62.9%** |

**Diagnostics:**

| Statistic | GT | Few-shot | LoRA |
|---|---|---|---|
| Semantic ratio | 21.1% | 11.3% | 37.4% |
| Segment count | 21.8 | 12.5 | 48.8 |
| Mean segment duration | 4.9s | 7.8s | 1.5s |
| Boundary count | 21.3 | 11.6 | 48.0 |

### Observations
- **Few-shot** is conservative: high accuracy (75%) from defaulting to beat, but misses most semantic boundaries (boundary recall 15.4%, semantic F1 19.8%). Under-predicts semantic ratio (11% vs GT 21%).
- **LoRA** over-segments: better boundary detection (boundary F1 35.1%, recall 62.9%) and slightly better semantic F1 (23.9%), but produces ~2× too many segments and over-predicts semantic (37% vs GT 21%).
- Neither planner significantly hurts downstream generation: few-shot FGD +0.020, LoRA FGD +0.026 vs GT labels — because the model's gesture-type conditioning has limited effect on the majority-beat spk2 distribution.
- LLM few-shot labels produce FGD 0.4575 vs. GT 0.4374 — only +0.020 degradation.
- LLM LoRA labels produce FGD 0.4634 vs. GT 0.4374 — +0.026 degradation, slightly worse than few-shot.
- Gesture-type conditioning is functional (all-semantic FGD=0.588 vs all-beat FGD=0.434), but since 62.6% of spk2 chunks are beat, the all-beat baseline is near-optimal. LLM planners that over-predict semantic (LoRA: fine-grained per-word labels) incur a small penalty (+0.026 FGD vs GT), while few-shot coarser labels only add +0.020.
- LoRA planner generates per-word annotations (~42 lines/recording = 21 seg pairs) vs. few-shot phrase-level (~6 lines/recording = 3 seg pairs). The fine-grained LoRA labels do not improve over coarser few-shot labels.
- BC is higher for LoRA (0.713) vs. GT (0.694) and fewshot (0.691) — more semantic gesture labels may slightly improve beat alignment despite worse FGD.

---

## Per-Speaker FGD (Allspk Baseline, g=0.5, seed=42)

**Model:** `e2e_perceptual/checkpoints_moment_allspk_clean/best.pt` + `vae_temporal_lean/best.pt`
**Script:** `run_per_speaker_fgd_single.py` (single-pass: loads VAE + prior once, iterates all speakers)

| Speaker | FGD ↓ | BC ↑ | L1Div ↑ | n |
|---|---|---|---|---|
| spk 1 | 1.7149 | 0.4737 | 8.42 | 15 |
| spk 2 | 0.5822 | 0.6876 | 15.71 | 15 |
| spk 3 | 0.9161 | 0.5141 | 10.58 | 15 |
| spk 4 | 0.6802 | 0.5416 | 13.19 | 15 |
| spk 5 | 0.6188 | 0.3465 | 6.03 | 15 |
| spk 6 | 0.7346 | 0.4155 | 10.50 | 7 |
| spk 7 | 0.9545 | 0.3716 | 7.02 | 15 |
| spk 9 | 0.5360 | 0.3583 | 7.52 | 7 |
| spk10 | 0.8818 | 0.4249 | 9.18 | 15 |
| spk11 | **0.4407** | 0.2177 | 5.72 | 15 |
| spk12 | 0.9844 | 0.6782 | 13.75 | 9 |
| spk13 | 1.0455 | 0.5095 | 9.42 | 9 |
| spk15 | 0.7162 | 0.3957 | 8.57 | 9 |
| spk16 | 1.6007 | 0.5895 | 12.62 | 9 |
| spk17 | 0.7085 | 0.5265 | 9.86 | 9 |
| spk18 | 0.8885 | 0.5135 | 9.44 | 9 |
| spk20 | 0.8864 | 0.4705 | 10.07 | 9 |
| spk21 | 1.8095 | 0.4491 | 10.43 | 7 |
| spk22 | 0.9076 | 0.4611 | 8.70 | 9 |
| spk23 | **3.4143** | 0.4319 | 8.31 | 8 |
| spk24 | 0.6196 | 0.3372 | 6.44 | 9 |
| spk25 | 1.2496 | 0.1948 | 6.89 | 8 |
| spk27 | 1.2003 | 0.4573 | 11.54 | 9 |
| spk28 | 0.7448 | 0.2486 | 5.30 | 9 |
| spk30 | 0.8755 | 0.3343 | 7.92 | 9 |
| **Mean** | **1.028** | **0.438** | **9.32** | |
| **Median** | **0.886** | **0.449** | **9.18** | |

### Observations
- Highest FGD: spk23 (3.41), spk21 (1.81), spk1 (1.71) — likely speakers with distinctive or unusual motion styles that are harder to model with the allspk prior.
- Lowest FGD: spk11 (0.44), spk9 (0.54), spk2 (0.58) — these speakers have motion styles well-covered by the allspk training distribution.
- Allspk prior FGD on spk2 (0.5822) is substantially higher than the E2E spk2 FT result (0.4374), confirming that speaker-specific fine-tuning is beneficial.
- BC varies widely by speaker (0.19–0.69), reflecting differences in natural gesture-beat correlation across speakers.
- Per-speaker results saved in `per_speaker_fgd/` (one JSON per speaker + `all_speakers.json`).

---

## Leakage-Free Protocol

All checkpoint selection uses the **val split** exclusively:
- VAE `best.pt`: selected by val reconstruction loss (`split='val'` in `BEAT2PoseDataset`)
- Prior `best_ar.pt`: selected by val AR FGD (`get_test_files(..., split='val')`)
- E2E FT `best.pt`: selected by val AR FGD only (batch FGD never selects checkpoints)
- Guidance scale: swept on val set (`--split val`), applied to test set
- Final metrics: test set only, 5 seeds for mean ± std
