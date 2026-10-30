# Dynamic row-priority correction — 2026-10-30

The single queue is ordered by sigma1/sigma2 node, then optimizer. Idle GPUs claim the next search without a row barrier. Row completion wakes the background reporter; overlapping row spans are not idle-GPU estimates. Stable IDs and scientific settings are unchanged. Windows Python 3.13.5 / PyTorch 2.9.1+cpu: 124 tests discovered, 117 passed and seven POSIX-only skips in 238.351 seconds. Log: workspace timing-evidence/dynamic-rows-full.log. The blocked-first-row test proves later rows finish without waiting, all 140 IDs run once, and row callbacks occur only after their searches complete. Exact queue priority, pause/failure dispatch and background report notification pass. Bash syntax, compilation, design synchronization and whitespace checks pass. No GPU training was started; Linux CI is checked separately after push.

# Historical row-barrier release (superseded above) — 2026-10-30

Rows follow sigma1-major/sigma2-minor order. Each row waits for all four optimizer searches before advancing; job IDs, eight-candidate search, 2B schedule and input identity remain unchanged. Strict barriers introduce possible GPU idle time, so the old 10–15 day estimate is not newly certified.

- Windows Python 3.13.5 / PyTorch 2.9.1+cpu: full 121-test run passed (114 executed, seven POSIX-only skips) in 209.144 seconds. After final CSV/log presentation changes and a partial expert-interval ordering fix, all 22 targeted row/report/pool/smoke-gate tests passed in 17.360 seconds. The final suite contains 122 tests; Linux CI must certify the pushed commit. Logs: workspace `timing-evidence/row-progress-full.log` and `row-progress-final-focused.log`.
- Scheduling tests cover all six stable 140-job ranges, exact first row IDs, ascending node order, no next-row dispatch until all four tasks finish, pause and failure barriers, and partial expert intervals.
- Synthetic report tests verify incomplete candidates never fill final response cells, partial peer ranks stay missing, all four completed searches close a row, cohorts and run IDs are checked, falsely completed short budgets are rejected, and mixed evidence preserves the prior archive with an explicit error.
- Reporting runs automatically inside the pool: startup, every 300 seconds, row completion and controlled exit. ZIPs contain one generation of JSON, offline HTML and four CSV tables (35-row big table, long cell table, candidate progress and evaluation points); no weights are loaded or archived. Per-row archives are not overwritten. Live console summaries support platforms where runtime file download is unavailable.
- Report output is outside finalized scope directories, so periodic publication cannot invalidate scope manifests. Live observations explicitly identify their generation/time and are not transactional checkpoints. Candidate progress distinguishes observed steps from durable checkpoint steps and labels the diagnostic step.
- Python compilation, Bash syntax, authoritative design synchronization and whitespace checks passed. Linux TERM/SIGKILL/lease recovery and 2/4-rank Gloo checks remain in CI; actual ACP/H100 report I/O overhead, row idle cost, save grace period and shared-filesystem behavior still require hardware acceptance. No GPU training or real input preparation was started.

---

# v1.8 platform stop/restart validation — 2026-10-29

The primary ACP workflow is now a batch job: submit the formal startup command, stop through the platform, and start the same command with the same persistent volumes and frozen software/input identity.

- Windows Python 3.13.5 / PyTorch 2.9.1+cpu: 113 tests discovered, **106 passed, 7 POSIX-only tests skipped**, in 183.284 seconds. Log: workspace `timing-evidence/platform-restart-windows.log`. Python compilation, all Bash syntax, design synchronization and whitespace checks passed.
- New POSIX tests run under Linux CI: nonblocking lease exclusivity, stable lease inode, abandoned owner recovery across container hostnames, legacy/malformed marker refusal, unsupported-filesystem refusal, and removal of only recognized atomic temporary files.
- Two Linux subprocess integrations use the real Bash formal supervisor and real tiny CPU training/checkpoints. Only H100 dispatch is replaced by a CPU fixture. TERM is sent solely to the outer supervisor PID and must reach training, save a positive-step checkpoint, wait for exit and skip finalization. A separate SIGKILL of the owned test process group leaves owner markers; restarting the same command must reacquire the kernel leases, recover and produce model/optimizer states identical to uninterrupted reference training.
- Persistent `.lease` files are excluded from scope delivery manifests. A finalized scope startup verifies existing artifacts without launching another pool. Existing data preparation, search replay, retention, four-optimizer resume and optional pause/resume regression coverage is retained.
- Linux CI results must be checked for the release commit; Windows skips are not evidence of Linux signal correctness. H100/ACP stop grace period, real save duration and shared-filesystem cross-container flock semantics still require a short target-platform acceptance run.
- The external `/data/260010171/run_v18_ready.sh` was not provided and is outside this repository. The documented platform command execs directly into the maintained ACP entrypoint. No actual GPU training or real corpus preparation was started.

---

# v1.8 resume maintenance validation — 2026-10-28

The earlier 95-test release missed search-level checkpoint retention on restart. An isolated reproduction showed replay deleting a pending candidate's latest checkpoint and a later final winner. This maintenance change limits pruning to already replayed terminal candidates; it does not change scientific parameters or require regenerating registered v1.8 inputs.

- Windows Python 3.13.5 / PyTorch 2.9.1+cpu: final complete **105-test suite passed in 138.815 seconds**. Log: workspace `timing-evidence/resume-release-tests.log`.
- New tiny-model tests exercise all four optimizers with candidate 1 paused after candidate 0 completed: pending step survives replay, recovered search decisions/loss and final tensors equal uninterrupted execution, and repeated completed-search startup preserves winner bytes. A separate forced later-winner case protects candidate 7 independently of incidental tiny-model ranking.
- Cooperative pause tests verify saved-step reuse and exact Muon model/optimizer-state equality after resume. Rollback tests discard uncommitted diagnostics above the saved step; a missing checkpoint after positive progress raises instead of silently restarting.
- Pool tests cover stopping dispatch even if a current job completes successfully, refusing child creation after stop, holding the launcher lock before clearing a pause request, rejecting an ordinary launch of a paused scope, and a nonzero paused exit.
- Bash wrapper tests verify pause needs no Python/CUDA environment, waits on outstanding locks without deleting them, and resume forwards the explicit resume flag. Intermediate full runs detected a new test fixture overriding the requested four-GPU visibility; the helper argument was corrected, all 10 ACP tests passed, then the final complete suite above passed. No production GPU-count check was weakened.
- All Bash syntax checks, Python compilation, authoritative design synchronization and whitespace checks passed. Bash-backed tests ran outside the Windows sandbox due to MSYS initialization restrictions.
- Real H100 save duration, process termination and pause/resume behavior remain hardware acceptance checks; CPU tests and Linux/Gloo CI are implementation evidence. No real GPU training, corpus preparation or long experiment was started.
- Keep the same commit/environment/input/micro/output for a trajectory. Existing v1.8 inputs remain usable, but updating from the previous commit requires fresh smoke outputs in a new RUN_BASE before formal training. Previously deleted checkpoints cannot be recovered by this fix.

---

# v1.8 implementation validation — 2026-10-28

The input workflow now prepares a fresh bundle on the target server. The 35-node, eight-candidate, 2B-token budget and last1-first execution are unchanged. Records below this section describe earlier releases, including the removed migration route.

- Windows Python 3.13.5 / PyTorch 2.9.1+cpu: all 95 tests passed in 80.970 seconds. Fifteen focused input/protocol/smoke-gate tests also passed before the complete suite.
- Fresh-preparation tests cover empty-root ownership, rejection and preservation of existing files, interrupted preparation retry, concurrent-writer rejection, immutable registered-bundle reuse, and rejection of old or migrated input identities.
- A mocked dataset stream verifies the pinned source revision, streaming/shuffle settings, 10M validation then 1B training quotas and fresh-source metadata. It does not download the real corpus. Tiny CPU integration tests separately cover freezing, registration, tamper detection and optimizer/resume behavior.
- ACP routing tests reject the removed migration command before installation or training. Current input routing, smoke and formal-launch regression tests pass using process stubs.
- All Bash scripts passed syntax checks; Python compilation and authoritative design/snapshot synchronization passed. Bash-backed tests ran outside the Windows sandbox because MSYS initialization is unavailable inside it.
- Production Hugging Face access, full preparation duration/cache peak, H100 BF16 smoke, actual storage usage and four-GPU throughput remain target-server checks. No real corpus preparation or GPU training was started for this change.
- CI results for this release are recorded by the pushed commit's GitHub Actions run; the older evidence below does not certify v1.8.

---

# v1.7 implementation validation — 2026-10-28 (historical)

Current release applies 35 nodes, five initial plus three adaptive LR candidates, and a true 3815-step / 2,000,158,720-token schedule to every scope. The current operator entrypoint defaults to last1 and never chains scopes.

- Windows Python 3.13.5 / PyTorch 2.9.1+cpu: full 92-test suite passed in 78.016 seconds. After final review corrected the smoke validator's CLI exponent representation (0.0 versus 0 in JSON identity), all nine v1.7 regression tests passed again in 2.534 seconds with a fixture matching the real argparse float.
- Migration tests verify source files remain unchanged, corpus and initialization bytes are identical, training windows are a strict prefix, evaluation windows are identical, lineage and registration are valid, and reused/overlapping/tampered destinations or sources are rejected. These use tiny CPU fixtures, not a newly migrated production server bundle.
- Direct training rejects old frozen budgets and inconsistent configs for all six scopes. Protocol tests cover every job's common budget, 3815 steps, 21 evaluation events, warmup and cosine endpoint.
- Operator tests cover last1 jobs 140–279, all four optimizer smoke candidates, wrong-scope/missing smoke and input changes, registration requirements, migration routing, and no finalization after pool failure. Shell tests use process stubs, not real GPUs.
- All Bash scripts passed syntax checks; Python compile checks, 840-job dry manifest and authoritative design/snapshot comparison passed. Optional pilot dry-run requires no inputs and creates no run directory.
- Git Bash could not initialize MSYS named objects inside the Windows sandbox; the full suite and Bash checks were run successfully outside that sandbox. One intermediate run detected files changing during an identity test; it was repeated with fixed files. No assertion was weakened to accept a changed identity.
- H100 BF16 execution, real four-GPU concurrency/throughput, production input migration, and real checkpoint/disk sizes still require the delivered target-server smoke and first jobs. No GPU or long training was started during this implementation.
- CI status for the pushed commit is separate from local verification; earlier CI records below do not certify v1.7.

---

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
