# FlowerLite-L: Implementation Context

## 1. Objective

Build and evaluate a unique lightweight classifier named **FlowerLite-L** for CIFAR-10 and CIFAR-100. The falsifiable hypothesis is: **under the same data split and training budget, FlowerLite-L trained with the selected SGD recipe will outperform its plain residual baseline on validation Top-1 accuracy while remaining below 6,000,000 learnable parameters and below 1,000,000,000 FLOPs per 32×32 image**. The hypothesis is confirmed only if the mean gain across seeds 2026, 2027, and 2028 is at least 0.30 percentage points on CIFAR-100, neither dataset regresses by more than 0.10 points, and all resource constraints pass.

The exam requires experiments on both datasets, multiple initial learning rates, SGD and Adam, complete training settings, a PDF report, and a non-duplicated architecture. Accuracy ranking is important, but test data must never be used for model selection.

## 2. Assumption register

| ID | Missing specification | Working assumption | Rationale | Blast radius if wrong |
|---|---|---|---|---|
| A1 | FLOP convention | Report MACs and FLOPs separately; define 1 MAC = 2 FLOPs for the hard gate | Counting tools disagree | A different instructor convention changes the reported number, not eligibility because both estimates remain below 1G |
| A2 | Compute platform | Kaggle T4 x2 (16 GB per GPU), Kaggle-like sessions, 20–30 total GPU-hours | Conservative student-accessible target | Batch size and total sweep scope may change |
| A3 | Pretraining | Train from scratch; no external data or pretrained weights | Fair CIFAR comparison and easy oral defense | If pretraining is allowed, ranking strategy may change |
| A4 | Test-time augmentation | Disabled in the primary score; optional horizontal-flip TTA reported separately only after approval | Avoid protocol ambiguity | Primary ranking may leave a small gain unused |
| A5 | Validation | Fixed stratified 45,000/5,000 development split from official training set | Protect official test set | Final retraining logic depends on this split |
| A6 | Codebase | Greenfield PyTorch repository using the structure in this file | No baseline repository was supplied | Paths must be mapped if integrating into an existing repository |
| A7 | Adam meaning | Use `torch.optim.Adam`, not AdamW | The exam explicitly names Adam | Weight-decay behavior differs from AdamW |
| A8 | Final checkpoint | Select epoch on validation, then retrain on all 50,000 train images for that fixed epoch count | No validation remains in final fit | If the instructor expects checkpoint selection on test, do not comply; ask for clarification |

## 3. Blocking issues

| ID | Issue | Why it blocks | Required resolution |
|---|---|---|---|
| B1 | Architecture-duplication rule has no registry of other groups | Uniqueness cannot be proven locally | Submit the architecture fingerprint in Section 5 to the instructor or class registry before expensive final runs |
| B2 | Official ranking evaluation command is not provided | Exact score protocol cannot be reproduced | Add the provided command without changing preprocessing or checkpoint semantics |
| B3 | GPU/session limits are assumed | Runtime estimates are not measured | Run `benchmark.py` before launching the full sweep and update only runtime planning, not experiment semantics |

No implementation may silently resolve B1–B3. Record the resolution in `experiments/CHANGELOG.md`.

## 4. Mathematical formulation

### 4.1 Notation

| Symbol | Shape/domain | Meaning |
|---|---|---|
| `x` | `[B,3,32,32]`, float32/float16 | Normalized image batch |
| `y` | `[B]`, int64 | Hard class indices |
| `Y` | `[B,K]`, float32 | One-hot or mixed target distributions |
| `K` | `{10,100}` | Number of classes |
| `C_s` | `{64,128,256,384}` | Stage widths |
| `D_s` | `{3,4,6,3}` | Blocks per stage |
| `z` | `[B,K]`, float32 | Output logits |
| `p` | `[B,K]`, float32 | Softmax probabilities |
| `lambda` | `[0,1]` | MixUp/CutMix mixing coefficient |
| `q` | `[B,N,h,d]` | Attention queries; `N=64,h=4,d=64` |

### 4.2 Stem

For input `x`, the stem is

`h_0 = SiLU(BN(Conv3x3_{3→64}(x)))`. **(Eq. 1)**

The convolution has stride 1 and padding 1. No max-pooling is allowed because the source image is only 32×32.

### 4.3 PetalMix residual block

For block input `h` with `C_in` channels, output width `C_out`, and stride `s`, first apply pre-activation:

`u = SiLU(BN(h))`. **(Eq. 2)**

Split channels contiguously into `u_a` and `u_b`, with `C_a=floor(C_in/2)` and `C_b=C_in-C_a`. Apply depthwise multi-scale filtering:

`v = Concat(DWConv3x3_s(u_a), DWConv5x5_s(u_b))`. **(Eq. 3)**

Apply deterministic two-group channel shuffle `S_2`, expansion, and activation:

`e = SiLU(BN(Conv1x1_{C_in→2C_out}(S_2(v))))`. **(Eq. 4)**

Squeeze-excitation with reduction ratio 8 is

`g = sigmoid(W_2 SiLU(W_1 GAP(e)))`,

`e_se = e ⊙ reshape(g,[B,2C_out,1,1])`, **(Eq. 5)**

where hidden width is `max(16, floor(2C_out/8))`. Project back:

`r = BN_0(Conv1x1_{2C_out→C_out}(e_se))`, **(Eq. 6)**

where `BN_0.weight` is initialized to zero and bias to zero. The shortcut is identity when `s=1` and `C_in=C_out`; otherwise it is `BN(Conv1x1_s(h))`. With stochastic depth `DropPath`,

`h_out = shortcut(h) + DropPath(r)`. **(Eq. 7)**

All operations use standard PyTorch autograd. Channel splitting and shuffle are permutations, so backward is the inverse permutation. No custom backward or CUDA extension is permitted.

### 4.4 Pollen attention bridge

After stage 3, `a∈R^{B×256×8×8}` is flattened to `t∈R^{B×64×256}` and layer-normalized. With four heads and per-head dimension 64:

`q,k,v = reshape(W_qkv LN(t))`,

`A = softmax((q k^T)/sqrt(64) + R)`,

`t_attn = t + DropPath(W_o reshape(A v))`. **(Eq. 8)**

`R∈R^{4×64×64}` is indexed from a learned 2-D relative-position-bias table of shape `[4,225]`, corresponding to offsets on an 8×8 grid. Then use a convolutional local feed-forward network:

`t_out = t_attn + DropPath(flatten(Conv1x1_{384→256}(SiLU(BN(DWConv3x3(Conv1x1_{256→384}(unflatten(LN(t_attn)))))))))`. **(Eq. 9)**

The authoritative LocalFFN parameterization is `256→384→256` with depthwise 3×3 at 8×8. Total bridge parameters must be measured. If the full model exceeds 5.8M parameters, halt rather than silently shrinking widths.

### 4.5 Classifier

After stage 4, global average pooling gives `g∈R^{B×384}`. The head is

`z = W_2 Dropout_{0.15}(SiLU(LN(W_1 g)))`, **(Eq. 10)**

with `W_1:384→768` and `W_2:768→K`.

### 4.6 Targets and loss

MixUp uses

`x~=lambda x_i + (1-lambda)x_j`,

`Y~=lambda onehot(y_i)+(1-lambda)onehot(y_j)`, `lambda~Beta(alpha,alpha)`. **(Eq. 11)**

CutMix replaces rectangle `M` and corrects lambda using the realized area:

`x~=M⊙x_i+(1-M)⊙x_j`,

`lambda~=1-|1-M|/(H W)`,

`Y~=lambda~ onehot(y_i)+(1-lambda~)onehot(y_j)`. **(Eq. 12)**

For soft target `Y` and label smoothing `epsilon`, define `Y'=(1-epsilon)Y+epsilon/K`. The objective is

`L=-B^{-1} sum_i sum_c Y'_{ic} log_softmax(z_i)_c`. **(Eq. 13)**

Do not apply label smoothing twice. If the mixed-target implementation already smooths targets, `CrossEntropyLoss(label_smoothing=0)` must be used.

### 4.7 Optimization

SGD update:

`m_t=0.9m_{t-1}+grad L(theta_t)`,

`theta_{t+1}=theta_t-eta_t(m_t+wd·theta_t)`. **(Eq. 14)**

Use Nesterov momentum in code. Adam is the exact PyTorch Adam update with betas `(0.9,0.999)`, epsilon `1e-8`, and configured L2 weight decay. The learning rate has five-epoch linear warm-up and cosine decay:

`eta_t=eta_max·(t/T_w)` for `t<T_w`, otherwise

`eta_t=eta_min+0.5(eta_max-eta_min)(1+cos(pi(t-T_w)/(T-T_w)))`. **(Eq. 15)**

EMA is updated after every optimizer step:

`theta_EMA←0.9998 theta_EMA+0.0002 theta`. **(Eq. 16)**

## 5. Architecture specification

### 5.1 Canonical configuration

```yaml
model:
  name: flowerlite_l
  stem_width: 64
  stage_widths: [64, 128, 256, 384]
  stage_depths: [3, 4, 6, 3]
  expansion: 2.0
  se_ratio: 0.125
  activation: silu
  drop_path_rate: 0.10
  attention_after_stage: 3
  attention_heads: 4
  attention_head_dim: 64
  attention_mlp_dim: 384
  head_hidden_dim: 768
  head_dropout: 0.15
```

### 5.2 Architecture fingerprint

`FL-L:S3x3-64|PM[d3,w64,s1]-PM[d4,w128,s2]-PM[d6,w256,s2]-PA[h4,d64,m384,8x8]-PM[d3,w384,s2]|H384-768-K|DW(3,5)+Shuffle2+SE8`

This exact string must be printed by `tools/model_info.py`, stored in every run manifest, and submitted early for uniqueness checking.

### 5.3 Initialization

- Convolution and linear weights: Kaiming normal for SiLU, except QKV/projection and classifier linear layers use truncated normal std 0.02.
- BN scale 1 and bias 0, except Eq. 6 terminal BN scale 0.
- LayerNorm scale 1 and bias 0.
- Relative-position bias table: truncated normal std 0.02.
- Classifier bias: 0.

## 6. Tensor flow

```text
image [B,3,32,32] f32/f16
  │
  ▼
Stem Conv3x3+BN+SiLU (Eq.1)
  │ [B,64,32,32]
  ▼
Petal Stage 1: 3 blocks (Eq.2–7)
  │ [B,64,32,32]
  ▼
Petal Stage 2: 4 blocks, first stride 2
  │ [B,128,16,16]
  ▼
Petal Stage 3: 6 blocks, first stride 2
  │ [B,256,8,8]
  ▼
Pollen Attention: 64 tokens, 4 heads (Eq.8–9)
  │ [B,256,8,8]
  ▼
Petal Stage 4: 3 blocks, first stride 2
  │ [B,384,4,4]
  ▼
Global Average Pool
  │ [B,384]
  ▼
LN → Linear 384→768 → SiLU → Dropout → Linear 768→K (Eq.10)
  │
  ▼
logits [B,K] f32
```

## 7. Tensor shape contract

| Name | Shape | Dtype/device | Constraint |
|---|---|---|---|
| `image_uint8` | `[B,3,32,32]` | uint8 CPU | values 0–255 |
| `x` | `[B,3,32,32]` | f32 CPU then f16/f32 CUDA | normalized by fixed dataset statistics |
| `label` | `[B]` | int64 CUDA | `0≤label<K` |
| `stem` | `[B,64,32,32]` | AMP dtype CUDA | finite |
| `s1` | `[B,64,32,32]` | AMP dtype CUDA | finite |
| `s2` | `[B,128,16,16]` | AMP dtype CUDA | finite |
| `s3` | `[B,256,8,8]` | AMP dtype CUDA | finite |
| `tokens` | `[B,64,256]` | AMP dtype CUDA | row-major spatial flattening |
| `attn_logits` | `[B,4,64,64]` | float32 CUDA | softmax computed in float32 |
| `attn_prob` | `[B,4,64,64]` | float32 CUDA | rows sum to 1 within 1e-5 |
| `s4` | `[B,384,4,4]` | AMP dtype CUDA | finite |
| `embedding` | `[B,384]` | AMP dtype CUDA | finite |
| `logits` | `[B,K]` | float32 CUDA | unnormalized, finite |
| `soft_target` | `[B,K]` | float32 CUDA | each row sums to 1 within 1e-6 |
| `loss` | scalar | float32 CUDA | finite and non-negative |

## 8. Numerical stability

| Operation | Requirement |
|---|---|
| Attention scaling/softmax | Cast logits to float32 before scaling and softmax; subtract row maximum implicitly via PyTorch softmax |
| Adam epsilon | Exactly 1e-8 |
| AMP | Use CUDA autocast float16 and GradScaler; loss, metrics, target mixing, softmax, and EMA master weights remain float32 |
| Soft-target log-softmax | Compute in float32 |
| BatchNorm | Keep module parameters and running stats float32 |
| Gradient clipping | Clip global norm at 5.0 after unscale; log pre-clip norm |
| Input | Assert shape `[B,3,32,32]`, finite values after normalization |
| Loss | Abort run on first non-finite loss; save diagnostic batch indices and config |
| Attention | Assert row sums absolute error below 1e-5 in debug mode |
| Mix labels | Assert row sums absolute error below 1e-6 and range `[0,1]` |

`torch.autograd.set_detect_anomaly(True)` is allowed only in the tiny overfit/debug command and forbidden in benchmark/full training due to overhead.

## 9. Resource budget

The authoritative numbers are produced by `python tools/model_info.py`; estimates below are planning values and must never be reported as measured results.

| Item | Planning value | Hard condition |
|---|---:|---:|
| Learnable parameters, CIFAR-10 | approximately 5.385M (derived planning value) | `<6.0M` |
| Learnable parameters, CIFAR-100 | approximately 5.454M (derived planning value) | `<6.0M` |
| MACs at batch 1 | approximately 0.284G (derived planning value) | Report measured |
| FLOPs at 2 FLOPs/MAC | approximately 0.568G (derived planning value) | `<1.0G` |
| Peak VRAM, batch 256 AMP | unmeasured | `<14.0 GB` per GPU on 16 GB T4 |
| Disk per run | unmeasured | `<2.0 GB`, keeping best and last checkpoints |

Parameter memory alone is roughly `5.454M×4=21.816 MB` in fp32. With fp32 weights, gradients, SGD momentum, EMA, and fp16 model copies, persistent model-state memory is roughly `21.816×4+10.908=98.172 MB`; activations dominate. This is a derived planning value until measured.

Pinned minimum environment:

```text
Python 3.10 or 3.11
PyTorch 2.4.x
Torchvision 0.19.x
CUDA runtime compatible with the installed PyTorch wheel
numpy 1.26.x
PyYAML 6.0.x
pandas 2.2.x
scikit-learn 1.5.x
matplotlib 3.9.x
seaborn 0.13.x
fvcore 0.1.5.post20221221
pytest 8.3.x
ruff 0.6.x
```

No custom CUDA extension is used.

## 10. Data protocol

Use `torchvision.datasets.CIFAR10` and `CIFAR100` official splits. Development mode creates a stratified 45,000/5,000 split from official training data with seed 2026 and saves ordered integer indices and SHA-256. Official test labels are inaccessible to the training and tuning code path.

Training transform order:

1. RandomCrop 32 with padding 4 and reflect padding.
2. RandomHorizontalFlip with probability 0.5.
3. TrivialAugmentWide, enabled by config.
4. ToImage/ToDtype float32 scaled to `[0,1]`.
5. Normalize using fixed dataset-specific mean/std stored in config.
6. Batch-level MixUp or CutMix after collation, never both on the same sample.

Validation/test transform: ToImage, ToDtype float32, Normalize only. No resize or random operation.

Normalization constants must be versioned in config and tested; changing them creates a new dataset protocol and a changelog entry.

## 11. Experiment matrix

Mandatory fair matrix, each with seed 2026 and 120 proxy epochs:

| Dataset | Optimizer | Initial LR |
|---|---|---:|
| CIFAR-10 | SGD Nesterov | 0.10 |
| CIFAR-10 | SGD Nesterov | 0.15 |
| CIFAR-10 | Adam | 0.001 |
| CIFAR-10 | Adam | 0.002 |
| CIFAR-100 | SGD Nesterov | 0.10 |
| CIFAR-100 | SGD Nesterov | 0.15 |
| CIFAR-100 | Adam | 0.001 |
| CIFAR-100 | Adam | 0.002 |

All other settings must be identical within a dataset. Select the optimizer/LR by validation Top-1. Run the winner at seeds 2026, 2027, and 2028 for 300 epochs on CIFAR-10 and 400 epochs on CIFAR-100. Then retrain once on all 50,000 official training images using the median best epoch from the three development seeds.

## 12. Evaluation

Primary metric: Top-1 accuracy = `correct predictions / evaluated examples × 100`. Secondary metrics: Top-5 for CIFAR-100, macro-F1, negative log-likelihood, expected calibration error with 15 equal-width bins, parameter count, MACs, FLOPs, throughput, and peak VRAM.

Evaluation must use `model.eval()`, `torch.inference_mode()`, deterministic transform, and float32 logits. Save predictions and labels as NPZ, confusion matrix PNG, per-class CSV, and `metrics.json`. No cherry-picking: every run appends one row to `experiments/results.csv`.

Acceptance tolerances:

- Re-evaluating one checkpoint must reproduce Top-1 within 0.02 percentage points.
- Number of evaluated examples must equal 5,000 for development validation or 10,000 for official test.
- Confusion-matrix sum must equal evaluated-example count.
- Macro-F1 implementation must agree with scikit-learn within 1e-10 on a fixed synthetic unit-test vector.

## 13. Results table shell

| Model | Params M | FLOPs G | Optimizer | LR | Epochs | CIFAR-10 Top-1 mean±std | CIFAR-100 Top-1 mean±std |
|---|---:|---:|---|---:|---:|---:|---:|
| Plain residual baseline | [MEASURE] | [MEASURE] | SGD | [SELECT] | [SELECT] | [MEASURE] | [MEASURE] |
| FlowerLite-L, no attention | [MEASURE] | [MEASURE] | SGD | [SELECT] | [SELECT] | [MEASURE] | [MEASURE] |
| FlowerLite-L, no multi-scale | [MEASURE] | [MEASURE] | SGD | [SELECT] | [SELECT] | [MEASURE] | [MEASURE] |
| FlowerLite-L, full | `<6.0` | `<1.0` | SGD/Adam winner | [SELECT] | 300/400 | [MEASURE] | [MEASURE] |
| FlowerLite-L, full + flip TTA | same | [MEASURE] | same | same | same | [OPTIONAL] | [OPTIONAL] |

Success targets are project gates, not promised outcomes: validation Top-1 at least 94.5% on CIFAR-10 and 75.0% on CIFAR-100 by epoch 120; final targets at least 96.5% and 80.0%, respectively. Report actual values even if lower.

## 14. Repository contract

```text
configs/{base,cifar10,cifar100,sweeps,ablations}/
src/flowerlite/{config.py,data.py,augment.py,models/,losses.py,optim.py,engine.py,metrics.py,tracking.py,utils.py}
tools/{model_info.py,benchmark.py,make_split.py,aggregate.py,render_report.py}
train.py
evaluate.py
tests/
experiments/results.csv
experiments/CHANGELOG.md
reports/final_report.md
requirements.txt
README.md
```

Every run directory contains resolved config, manifest, metrics history, final metrics, stdout log, best and last checkpoints, predictions, and figures. Run IDs are immutable: `fll_YYYYMMDD-HHMM_dataset-change-seed`.

## 15. Out of scope

- External data, pretrained weights, knowledge distillation from a model above the limits, pseudo-labeling the test set, or test-set tuning.
- Neural architecture search, hyperparameter optimization frameworks, or more than the declared matrix without human approval.
- SAM in the default plan; its doubled backward cost conflicts with the assumed budget.
- Model ensembles for the primary score.
- Changing official labels, class mapping, train/test split, or metric definition.
- Custom CUDA kernels.
- Quietly reducing batch size, epochs, widths, depths, or validation size to make a run pass.
- Claiming a leaderboard rank before the official leaderboard exists.
