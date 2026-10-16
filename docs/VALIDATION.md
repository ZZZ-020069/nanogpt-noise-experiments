# Implementation validation — 2026-10-16

These checks are implementation tests, not research results.

## Passed locally

- Windows, Python 3.13.5, PyTorch 2.9.1+cpu, NumPy 2.1.3.
- `python -m unittest discover -s tests -v`: 61 tests passed, including the core protocol, launcher/preflight, independent-GPU pool, scope isolation, retention and server-side input-registration tests.
- Exact 124,373,760 full-model and 7,079,424 × k trainable parameter counts; tied weights and frozen scopes.
- Muon formula and 0.03 auxiliary binding; Adam/AdamW/SGD decay; LR-zero optimizer-state update.
- Noise RMS, clean behavior, independent RNG and missing-gradient errors.
- Initial five LR points, upper/lower extension, tie/failure midpoint, all-failed search, eight unique candidates.
- CPU small-model four-optimizer training: interrupted/resumed weights exactly equal uninterrupted weights.
- Small-model eight-candidate node controller and idempotent completed-node restart.
- Numerical failure terminal state, failed-step token accounting, finite-large-loss completion, incompatible resume rejection.
- Data identity tamper detection and missing response values left missing.
- Server-side input registration binds the corpus, initialization and fixed windows to one input ID; a changed registered input is rejected after smoke. Token-boundary truncation, immutable registration, completed-bundle reuse and ACP input-mode routing have dedicated tests.
- Actual fixed corpus metadata/size/token-range/hash checks; common 124M initialization and all window indices generated.
- Python compile checks, Bash syntax checks, design snapshot synchronization, complete 840-job dry manifest.
- Empty-result CSV/JSON and 30 heatmap rendering smoke checks (these blank images are not experimental results).
- Production launch dispatch tested with a stub executable: one-GPU smoke/formal isolation, preservation of scheduler GPU tokens, four independent workers, paths with spaces, invalid arguments and preflight error propagation. CUDA checks are mocked in unit tests, not hardware validation.
- The dynamic pool runs each job exactly once in a test interval, never overlaps two jobs on one GPU worker, and stops new assignments after the first child failure. A production failure also terminates active children so that the same interval can be resubmitted for checkpoint recovery.
- Smoke numerical failures now exit nonzero (2); infrastructure interruption remains 75. Formal numerical-failure accounting is unchanged.
- Terminal candidates losslessly compact per-step JSON into verified gzip JSONL; repeated compaction is idempotent. Each job retains only its current/final winner's single full checkpoint, including optimizer state.
- Named scope launch covers the exact 140-job ranges. Scope preparation rejects cross-scope output reuse; finalization checks completion and retention, produces a per-file SHA-256 manifest, downloaded copies detect size/hash tampering, and cleanup requires a finalized directory plus a scope-specific confirmation token.
- Static checkpoint accounting estimates decimal peak storage at 195.58 GB for `full`, 84.03 GB for `last1`, 90.43 GB for `last2`, 96.83 GB for `last3`, 116.03 GB for `last6`, and 154.43 GB for `last12`. These are implementation calculations, not target-filesystem measurements.

## Remote CI evidence

- The independent one-GPU-per-job pool change has passed the local CPU suite and Bash syntax checks but has not yet been validated by GitHub Actions or H100 hardware.

- The initial implementation commit `3c30469` passed Linux CPU unit tests and the original two-rank AdamW/Muon Gloo resume smoke in [Actions run 37866003505](https://github.com/ZZZ-020069/nanogpt-noise-experiments/actions/runs/37866003505).
- Commit `b65d8af` passed all 26 unit tests, Bash syntax validation, and both two-rank and four-rank Linux Gloo checks in [Actions run 38032717104](https://github.com/ZZZ-020069/nanogpt-noise-experiments/actions/runs/38032717104). Distributed coverage includes all four optimizers, full/last1 scopes, gradient accumulation, noisy gradients, bitwise checkpoint resume and a single-rank NaN failure propagated to all ranks.
- The first expanded run (`944097f`, Actions 38032523765) passed the 26 unit tests and two-rank checks, but failed the four-rank bitwise resume assertion. The default DDP reducer rebuilds gradient buckets after its first backward, so a newly constructed resumed reducer can change floating-point reduction order. The fix keeps the initial bucket layout using `find_unused_parameters=True, static_graph=False`; missing trainable gradients are still rejected. The rebuild condition is documented in [PyTorch's reducer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/reducer.hpp#L130-L135). The same exact-equality regression passed after the fix; it was not replaced by a loose tolerance. This is evidence for the tested CPU configuration, not a guarantee of bitwise CUDA reproducibility across hardware or software versions.

## Not yet validated

- H100 CUDA/BF16 kernels, memory peak, runtime/driver combination and throughput.
- Actual target-filesystem scope peak, filesystem metadata overhead, transfer duration and local-copy verification at full scale; the 195.58 GB `full` estimate has not yet been measured on H100 output.
- Target-server four-process independent CUDA execution and signal/restart behavior. NCCL is not required by the production pool because every training process has world size one; the retained DDP regression tests remain CPU/Gloo implementation coverage only.
- Local Windows Gloo: torchrun lacks libuv support; direct FileStore fallback reaches an unsupported Gloo device error. The Linux CI evidence above is separate from local execution.
- Full 1.01B-token preparation from the fixed Hugging Face revision on the target server, including its download duration and cache-space peak. The implementation pins preparation dependencies and records their realized versions, but the complete target-server bundle has not yet been generated.

No full-budget training, formal result, quadratic experiment or automatic Slurm submission was performed. Formal-launch readiness requires the target-hardware checks above.
