# FlowerLite-L Executable Tasks

## Global rules

- Execute tasks only in dependency order. Every dependency must be PASS.
- A failed verification blocks all downstream work. Never proceed on a red test.
- Phase gates are hard; never enter the next phase with an unmet gate.
- End every task as PASS, BLOCKED, or KILLED.
- Never use official test results for any decision.
- Every metric must be measured; use `[NOT MEASURED]` otherwise.
- Keep every command and raw terminal output in the run log.

## Dependency graph

```text
P0-01 → P0-02 → P0-03 → P0-04 → P0-05
                              └────→ P1-01 → P1-02 → P1-03 → P1-04
                                                     └────→ P2-01 → P2-02 → P2-03 → P2-04 → P2-05
                                                                                 └────→ P3-01 → P3-02 → P3-03 → P3-04 → P3-05
```

## Phase 0 — Environment and protocol verification

### TASK-ID: P0-01
**TITLE:** Bootstrap the pinned repository
**STOP/KILL CONDITION:**
PASS locally if CUDA works. HALT only if CUDA is unavailable or GPU ops fail.

### TASK-ID: P0-02
**TITLE:** Implement validated hierarchical configuration

### TASK-ID: P0-03
**TITLE:** Build protected CIFAR data pipeline

### TASK-ID: P0-04
**TITLE:** Implement metric and immutable tracking infrastructure

### TASK-ID: P0-05
**TITLE:** Record uniqueness disposition and platform benchmark

## Phase 1 — Cheapest disproof

### TASK-ID: P1-01
**TITLE:** Implement PetalMix and PollenAttention modules

### TASK-ID: P1-02
**TITLE:** Assemble FlowerLite-L and verify resource limits

### TASK-ID: P1-03
**TITLE:** Implement loss, optimizers, scheduler, EMA, and engine
**DESCRIPTION:**
Implement Eq. 13–16, AMP, gradient clipping, train/evaluate loops, checkpoint best/last, exact resume including RNG/dataloader state, CSV history, and development test-set lockout. Distributed training (DDP) must be supported: torchrun --standalone --nproc_per_node=2. Add tests for DDP.

### TASK-ID: P1-04
**TITLE:** Run tiny-overfit and short CIFAR-100 disproof
**VERIFICATION:**
Command: `torchrun --standalone --nproc_per_node=2 train.py --config configs/sweeps/cifar100_mve.yaml --stage all`

## Phase 2 — Full implementation and mandatory sweep

### TASK-ID: P2-01
**TITLE:** Generate the controlled eight-run sweep

### TASK-ID: P2-02
**TITLE:** Execute the mandatory proxy sweep
**VERIFICATION:**
Command: `torchrun --standalone --nproc_per_node=2 train.py --sweep configs/sweeps/mandatory.yaml --resume-incomplete`

### TASK-ID: P2-03
**TITLE:** Select winning optimizer and learning rate without test leakage

### TASK-ID: P2-04
**TITLE:** Run full three-seed final development training
**VERIFICATION:**
Command: `torchrun --standalone --nproc_per_node=2 train.py --config configs/cifar10/final_selected.yaml --seeds 2026 2027 2028 && torchrun --standalone --nproc_per_node=2 train.py --config configs/cifar100/final_selected.yaml --seeds 2026 2027 2028`

### TASK-ID: P2-05
**TITLE:** Freeze recipe and retrain on full official training sets
**VERIFICATION:**
Command: `torchrun --standalone --nproc_per_node=2 train.py --config configs/cifar10/submission.yaml --mode final && torchrun --standalone --nproc_per_node=2 train.py --config configs/cifar100/submission.yaml --mode final && python evaluate.py --config configs/cifar10/submission.yaml --split test && python evaluate.py --config configs/cifar100/submission.yaml --split test`

## Phase 3 — Ablations, report, and defense package

### TASK-ID: P3-01
**TITLE:** Implement fair architecture ablations

### TASK-ID: P3-02
**TITLE:** Run budget-aware CIFAR-100 ablations
**VERIFICATION:**
Command: `torchrun --standalone --nproc_per_node=2 train.py --sweep configs/ablations --resume-incomplete`

### TASK-ID: P3-03
**TITLE:** Generate final analysis artifacts

### TASK-ID: P3-04
**TITLE:** Build the final PDF report

### TASK-ID: P3-05
**TITLE:** Produce reproducibility and oral-defense package
