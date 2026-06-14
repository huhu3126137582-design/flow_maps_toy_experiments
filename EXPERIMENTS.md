# Experiment Notes

This document summarizes the training and evaluation protocol used for the
reported checkerboard experiments. The original chronological research log is
available in `方案.md`.

## Shared setup

- Target distribution: eight occupied cells in a 4 x 4 checkerboard on
  `[-1, 1]^2`.
- Time convention: `t=0` is data and `t=1` is standard Gaussian noise.
- Interpolation:

  ```text
  x_t = (1 - t) x_0 + t epsilon,  epsilon ~ N(0, I_2).
  ```

- Conditional velocity: `epsilon - x_0`.
- Flow-map parameterization:

  ```text
  F_theta(x, s, t) = x + (t - s) V_theta(x, s, t).
  ```

- Model: six hidden layers, hidden width 512, SiLU activations, and separate
  64-dimensional Gaussian Fourier embeddings for the start and target times.
- Optimizer: Adam.
- EMA decay: `0.999`.
- Directional derivatives: `torch.func.jvp`.
- Default training seed: `0`.
- Training and loss computation: FP32, with TF32 allowed on CUDA.

## Metrics

The project evaluates both checkerboard-specific structure and general
distribution agreement.

### Support metrics

- In-support rate: fraction of samples inside an occupied checkerboard cell.
- Distance to support: mean and 95th percentile distance for samples outside
  the occupied cells.
- Mode coverage: number of occupied cells receiving more than 1% of samples.
- Mode KL: divergence of occupied-cell proportions from the uniform
  distribution.
- Within-cell JSD: uniformity after dividing each occupied cell into an 8 x 8
  subgrid.

### Distribution metrics

- Histogram TV and JSD on a 96 x 96 grid over `[-1.2, 1.2]^2`, with an
  additional overflow bin.
- Sliced Wasserstein-2 using a fixed real-data reference and fixed projection
  directions.
- RBF MMD averaged over bandwidths `0.05`, `0.1`, `0.2`, and `0.5`.

NFE counts model evaluations. Each teacher Euler step costs one NFE, while
each equal-time flow-map segment costs one NFE.

## Stage 1: Flow Matching teacher

The teacher is trained only on the diagonal `s=t`, where its output represents
the instantaneous marginal velocity.

Default configuration:

| Setting | Value |
| --- | ---: |
| Steps | 120,000 |
| Batch size | 65,536 |
| Initial learning rate | 1e-3 |
| Final learning rate | 1e-4 |
| Gradient clipping | 10 |
| Checkpoint interval | 15,000 |
| Evaluation NFE | 128 |

The 120k EMA model was selected. On 50k samples it reached an in-support rate
of `0.99858`, histogram TV of `0.10465`, and histogram JSD of `0.00919`.

## Stage 2: Stop-gradient LMD

SG-LMD initializes the student from the frozen teacher. The off-diagonal target
is detached, while the JVP branch remains differentiable. A diagonal training
component preserves the instantaneous velocity boundary condition.

Default configuration:

| Setting | Value |
| --- | ---: |
| Main-run steps | 70,000 |
| Logical batch size | 32,768 |
| Default microbatch size | 2,048 |
| Initial learning rate | 3e-4 |
| Final learning rate | 3e-5 |
| Learning-rate schedule | Cosine decay over 90,000 steps |
| Gradient clipping | 5 |

Training starts with a 2k-step pilot. The diagonal probability and maximum
off-diagonal time gap increase during training. The final search deterministically
replays from step 60k to 75k and evaluates checkpoints from 65k through 75k.

Step 72k was selected by 1-NFE SW2 with in-support rate as the secondary
criterion. The final evaluation covers NFE values 1, 2, and 4, while the
visualizations additionally show NFE=8.

## Stage 3: Lagrangian self-distillation

LSD trains from data and noise without a frozen teacher. Diagonal examples use
the Flow Matching objective, while off-diagonal examples enforce
self-consistency through a detached mapped target and a differentiable JVP
branch.

Default configuration:

| Setting | Value |
| --- | ---: |
| Steps | 130,000 |
| Logical batch size | 100,000 |
| Default microbatch size | 2,048 |
| Diagonal fraction | 0.75 |
| Initial learning rate | 1e-3 |
| Gradient clipping | 10 |
| Late checkpoint interval | 5,000 |

Checkpoint selection first ranks candidates using NFE values 1, 2, 4, and 8.
The top candidates are then evaluated with seeds `20260614`, `20260615`, and
`20260616`. Each seed uses 50k generated samples and a 200k reference sample.
The final score is the equal-weight mean normalized rank across SW2,
histogram JSD, and in-support rate for all four NFE values.

Step 99k was selected. Its mean metrics across the three robustness seeds are
reported in the project README and in
`outputs/lsd/best_checkpoint_nfe_1_2_4_8_robustness.json`.

## Reproducibility

Training checkpoints include:

- model and optimizer state;
- EMA state;
- current step and elapsed time;
- Python, NumPy, PyTorch, and custom generator random states;
- the training configuration.

The tests cover flow-map identity, the diagonal derivative condition, JVP
values and gradients, detached target behavior, branch updates, time sampling,
ranking direction, and deterministic checkpoint recovery. CUDA recovery tests
are skipped on machines without CUDA.

The compact exported weights contain only the selected EMA model and metadata.
They are suitable for release assets, while full checkpoints are intended for
training recovery.
