# Implementation validation — 2026-10-08

These checks are implementation tests, not research results.

## Passed locally

- Windows, Python 3.13.5, PyTorch 2.9.1+cpu, NumPy 2.1.3.
- `python -m unittest discover -s tests -v`: 19 tests passed.
- Exact 124,373,760 full-model and 7,079,424 × k trainable parameter counts; tied weights and frozen scopes.
- Muon formula and 0.03 auxiliary binding; Adam/AdamW/SGD decay; LR-zero optimizer-state update.
- Noise RMS, clean behavior, independent RNG and missing-gradient errors.
- Initial five LR points, upper/lower extension, tie/failure midpoint, all-failed search, eight unique candidates.
- CPU small-model four-optimizer training: interrupted/resumed weights exactly equal uninterrupted weights.
- Small-model eight-candidate node controller and idempotent completed-node restart.
- Numerical failure terminal state, failed-step token accounting, finite-large-loss completion, incompatible resume rejection.
- Data identity tamper detection and missing response values left missing.
- Actual fixed corpus metadata/size/token-range/hash checks; common 124M initialization and all window indices generated.
- Python compile checks, Bash syntax checks, design snapshot synchronization, complete 840-job dry manifest.
- Empty-result CSV/JSON and 30 heatmap rendering smoke checks (these blank images are not experimental results).

## Not yet validated

- H100 CUDA/BF16 kernels, memory peak, runtime/driver combination and throughput.
- Linux NCCL multi-GPU execution and target-server signal/restart behavior.
- Linux Gloo two-process smoke: test and CI workflow provided, but not executed successfully locally. Windows torchrun lacks libuv support; direct FileStore fallback reaches an unsupported Gloo device error. No distributed pass is claimed.
- Online data re-preparation and exact reproduction across tokenizer/datasets versions. Existing corpus metadata does not preserve its original preparation dependency versions.
- GitHub Actions cannot be claimed green until the remote workflow actually runs.

No full-budget training, formal result, quadratic experiment or automatic Slurm submission was performed. Formal-launch readiness requires the target-hardware checks above.
