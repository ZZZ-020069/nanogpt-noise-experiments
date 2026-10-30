> Historical plan: the row barrier was superseded by 2026-10-30-dynamic-rows.md after user clarification.

# Row scheduling and automatic progress reports

**Goal:** Fill the 35-row response table in sigma1-major/sigma2-minor order, four optimizers per row, with reports generated inside the batch task.

**Architecture:** Preserve all job IDs and training budgets. Group pool jobs by scope/node and wait for all four optimizer searches before the next row. Publish lightweight JSON/CSV/HTML/ZIP snapshots outside the scope's immutable delivery tree, on a timer, at row completion and at supervisor exit. Training publishes small progress records without extra evaluations or checkpoint copies.

**Tech stack:** Python stdlib, existing atomic files and locks, Bash entrypoints, CPU fixtures and Linux CI.

- [ ] Add deterministic row batches and a row barrier; test first jobs 140/175/210/245, all 140 unique IDs, no next-row dispatch before prior row finishes, failure and stop behavior.
- [ ] Add per-candidate progress records with successful steps/tokens, durable checkpoint step, latest diagnostic metrics and runtime state; retain existing strict identities and rollback behavior.
- [ ] Add report publisher reading only metadata/status/progress/evaluation/search files. Only complete eight-candidate searches fill final table cells; incomplete results and partial peer ranks remain explicitly labeled. Validate code/data/environment cohorts.
- [ ] Publish latest report and one lightweight archive per completed row. Retain timestamps/generation IDs, mark live snapshots nontransactional, preserve missing values, exclude all model weights. Reporter failures are visible in logs and do not change training decisions.
- [ ] Start/stop reporting automatically inside the pool, without a terminal or extra ACP task. Preserve TERM/SIGKILL and same-command recovery. Closed report files stay outside the finalized scope tree.
- [ ] Update authoritative design, snapshot and operator docs, including row barrier idle cost and pending hardware validation. Run CPU/Bash tests and Linux CI; commit/push without real GPU training.
