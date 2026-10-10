# Implementation validation — 2026-10-10

These checks are implementation tests, not research results.

## Passed locally

- Windows, Python 3.13.5, PyTorch 2.9.1+cpu, NumPy 2.1.3.
- `python -m unittest discover -s tests -v`: 26 tests passed (19 core tests plus 7 launcher/preflight tests).
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
- Four-GPU Bash dispatch tested with a stub executable: smoke/formal separation, four workers, paths with spaces, invalid arguments and preflight error propagation. CUDA checks are mocked in unit tests, not hardware validation.
- Smoke numerical failures now exit nonzero (2); infrastructure interruption remains 75. Formal numerical-failure accounting is unchanged.

## Remote CI evidence

- The initial implementation commit `3c30469` passed Linux CPU unit tests and the original two-rank AdamW/Muon Gloo resume smoke in [Actions run 37866003505](https://github.com/ZZZ-020069/nanogpt-noise-experiments/actions/runs/37866003505).
- This change extends CI to two and four ranks, all four optimizers, full/last1 scopes, accumulation, noisy gradients, resume and a single-rank NaN failure. Its matching remote run must be checked after upload; the initial commit's green status is not evidence for the new test version.
- The first expanded run (`944097f`, Actions 38032523765) passed the 26 unit tests and two-rank checks, but failed the four-rank bitwise resume assertion. The default DDP reducer rebuilds gradient buckets after its first backward, so a newly constructed resumed reducer can change floating-point reduction order. The fix keeps the initial bucket layout using `find_unused_parameters=True, static_graph=False`; missing trainable gradients are still rejected. The rebuild condition is documented in [PyTorch's reducer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/reducer.hpp#L130-L135). The unchanged exact-equality regression is being rerun; it has not been replaced by a loose tolerance.

## Not yet validated

- H100 CUDA/BF16 kernels, memory peak, runtime/driver combination and throughput.
- Linux NCCL multi-GPU execution and target-server signal/restart behavior.
- Local Windows Gloo: torchrun lacks libuv support; direct FileStore fallback reaches an unsupported Gloo device error. The Linux CI evidence above is separate from local execution.
- Online data re-preparation and exact reproduction across tokenizer/datasets versions. Existing corpus metadata does not preserve its original preparation dependency versions.
- The expanded Linux two/four-rank CI checks are pending the new commit's remote run.

No full-budget training, formal result, quadratic experiment or automatic Slurm submission was performed. Formal-launch readiness requires the target-hardware checks above.
