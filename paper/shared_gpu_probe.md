# Shared-GPU ResNet feasibility trial

Completed 2026-10-17 at the user's request after both GPUs were found occupied.
This is a resource/performance check, not a scientific refinement experiment.

## Result

The short trial **passed** on GPU 0 while its existing Ray worker remained
active. All 100 batch-eight forward/backward/Adam-update steps completed with
finite losses and gradient norms. No out-of-memory error occurred.

| Measurement | Result |
| --- | ---: |
| Trial budget | 100 steps, 800 training examples |
| Timed steady-state steps | 90, after 10 warmups |
| Median / mean step time | 0.05955 / 0.06000 s |
| 90th-percentile step time | 0.06224 s |
| Timed throughput | 133.3 objects/s |
| Entire training-loop wall time | 16.02 s |
| Peak allocated / reserved PyTorch memory | 593.76 / 804 MiB |
| PyTorch allocator cap | 1,024 MiB |
| Peak sampled total process GPU memory | 1,308 MiB |
| Minimum sampled device free memory | 982 MiB |
| Console exit status | 0 |

The synchronized per-step timer includes host-to-device transfers, forward,
backward, finite-gradient checking and optimizer update. It excludes CPU data
loading/augmentation, W&B logging and GPU-status polling outside that interval.
The entire loop includes these costs and startup/warmup effects. Extrapolating
only the measured step work gives approximately **22.3 minutes for 22,270
updates (10 epochs)**. This is not an end-to-end training prediction: validation,
galleries, I/O and evolving contention add time.

## Configuration and limits

- Original epoch-49 ResNet (`legacy` head), its inherited Adam moments, linear
  representation, original valid-pixel L1, paired D4 augmentation, batch 8.
- Trial constant LR 1e-5; no training schedule or checkpoint selection tested.
- Only GPU UUID `GPU-7f523ed6-dcff-1d8d-5f3a-b4b8fcc4c3b0` was visible to the process.
- PyTorch allocator capped at 1 GiB; this does not cap total process memory or
  impose a compute-percentage limit. CUDA/context overhead explains the larger
  total process memory.
- Trial loop limited to 180 seconds; external timeout 300 seconds.
- Stop guards: at least 1,792 MiB device-free memory before initialization and
  stop if sampled remaining free memory falls below 256 MiB during the trial.
- No MPS server, Ray configuration or competing process was modified.
- The Ray worker's sampled allocation remained 43,156 MiB. Its application
  throughput was not measured, so this does **not** establish zero impact on it.
- The trial weights were discarded. No new scientific checkpoint was saved;
  the parent checkpoint SHA256 was rechecked and unchanged.
- Only one GPU and the original L1 objective were tested. Two simultaneous
  refinements and the source-balanced implementation remain untested.

After the process exited, GPU 0 returned to **43,165 MiB used / 2,296 MiB free**,
matching the pre-trial reading. Both original Ray processes were still present,
and the probe process was absent. GPU 1 was not used by this trial.

This demonstrates that sharing is feasible under the observed load despite
100% reported utilization. Available memory and throughput can change as the
other workloads change; longer jobs need the same monitoring and memory guard.
It does not reserve a fixed percentage of compute or memory for us.

## Artifacts

- Runner: `code/probe_shared_gpu.py`.
- Contract, source snapshots, selected training rows and measurements:
  `runs/shared-gpu-probe-v01/` (`report.json`, `contract.json`, `before.json`).
- Persistent session: `supernisp-shared-gpu-probe-v01` (finished, idle shell).
- Console: `runs/shared-gpu-probe-v01-console/console.log`; exit status `0`.
- [Online W&B trial](https://wandb.ai/AI-Astro/superNISP-corrected/runs/24xwfbn4).

The full paired refinement plan remains in
[resnet_refinement_handoff.md](resnet_refinement_handoff.md). Its new trainer and
comparison still require implementation; **no full refinement jobs are running
or queued**. GPU-sharing feasibility is no longer the immediate blocker.
