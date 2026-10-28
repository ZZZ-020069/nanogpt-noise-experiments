# Global 2B budget and last1-first implementation plan

> Execute inline in this session under the user's implementation and GitHub push authorization.

**Goal:** Publish v1.7 with the same 35-node, five-initial-plus-three-adaptive LR protocol for every scope, 3815 steps per candidate, and last1 as the current launch default.

**Architecture:** Keep one shared Budget and unchanged search/model/noise implementations. Derive a new registered input bundle from existing immutable corpus/initialization, taking the original training-window prefix. Enforce current protocol at both preflight and direct training. Run four optimizer smoke candidates for the requested scope before allowing its formal launch.

**Tech stack:** Python, PyTorch, unittest, Bash, Git.

## Tasks

- [x] Update the authoritative parent `nanogpt_experiment_design.md` to v1.7, synchronize `docs/nanogpt_experiment_design.md`, and distinguish earlier discussion/pilot records from current decisions.
- [x] Change `protocol.Budget.steps` to 3815 and VERSION to v1.7; test all six scopes, 840 jobs, 6720 candidate maximum, 21 evaluation events and 2,000,158,720 tokens.
- [x] Add `nanogpt_noise.migrate_inputs`: verified immutable source, exclusive new destination, byte-identical corpus and initialization, original windows prefix/evaluation windows, lineage, six-file registration. Reject edited, incompatible or reused destinations; test with CPU fixtures.
- [x] Reject noncurrent real input/config budgets in `train` and `preflight`; preserve explicit tiny CPU tests. Do not enable incompatible checkpoint resume.
- [x] Update `run_acp.sh`, `h100.sh`, `run_smoke.sh`, `run_formal.sh`, `run_4gpu.sh` and Slurm defaults. Smoke tests all four optimizers for last1 by default; formal launch runs only last1 and finalizes without chaining scopes. Bind smoke to code, input, scope and micro-batch.
- [x] Update README and RUN_TASK with separate existing-input migration/new-input preparation, smoke, launch, recovery and D-drive receipt instructions. Update current auxiliary pilot tools to a new 2B policy identity; preserve historical results and specifications as historical.
- [x] Run `python -m unittest discover -s tests -v` under CPU PyTorch, Bash syntax checks, design snapshot check, grid dry-run and diff review. GPU smoke remains for the user's H100.
- Delivery: commit reviewed source/docs/tests, push origin main without force, verify remote commit. Record the resulting SHA and remote verification in the delivery message; do not start GPU training.
