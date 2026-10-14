# Two-Command H100 Entrypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the normal server workflow to one smoke command and one formal-scope command while preserving the mandatory per-scope transfer and cleanup boundary.

**Architecture:** Add two small operator-facing Bash scripts over the already tested `scripts/h100.sh` lifecycle wrapper. `run_smoke.sh` performs every preflight and the eight-step GPU smoke; `run_formal.sh` runs or resumes one named scope and automatically finalizes it only after the four-GPU pool exits successfully. The six scopes remain separate because the design requires transfer, verification, and manual cleanup between scopes under the approximately 200 GB storage limit.

**Tech Stack:** Bash, Python 3.11, `unittest`, existing `nanogpt_noise` command-line modules.

---

### Task 1: Specify the two public commands

**Files:**
- Modify: `tests/test_operator.py`
- Test: `tests/test_operator.py`

- [ ] **Step 1: Add a failing smoke-shortcut test**

Add a test that invokes `scripts/run_smoke.sh` with one visible GPU and asserts that the smoke status and commit marker are produced.

- [ ] **Step 2: Add a failing formal-shortcut test**

Add a test that first completes smoke, invokes `scripts/run_formal.sh full` with four visible GPUs, and asserts that both `nanogpt_noise.pool` and `nanogpt_noise.scope finalize` are dispatched.

- [ ] **Step 3: Add a failure-boundary test**

Make the Python stub optionally fail `nanogpt_noise.pool`; assert that `run_formal.sh` returns nonzero and never calls finalize after a failed pool.

- [ ] **Step 4: Run the focused tests and verify failure**

```bash
python -m unittest tests.test_operator -v
```

Expected: the new tests fail because the two scripts do not exist.

### Task 2: Implement the shortcuts

**Files:**
- Create: `scripts/run_smoke.sh`
- Create: `scripts/run_formal.sh`
- Test: `tests/test_operator.py`

- [ ] **Step 1: Implement `run_smoke.sh`**

Resolve the repository root from the script location and delegate to `bash scripts/h100.sh smoke`. Do not duplicate validation or training logic.

- [ ] **Step 2: Implement `run_formal.sh`**

Accept zero or one argument, defaulting to `full`. Delegate first to `bash scripts/h100.sh scope "$scope"`; only after a zero exit code delegate to `bash scripts/h100.sh finalize "$scope"`. Print the finalized result path and the mandatory transfer/verification reminder.

- [ ] **Step 3: Run focused tests**

```bash
python -m unittest tests.test_operator -v
```

Expected: all operator tests pass.

### Task 3: Make documentation two-command-first

**Files:**
- Modify: `RUN_TASK.md`
- Modify: `README.md`

- [ ] **Step 1: Rewrite the short task guide**

Present only these normal commands after the one-time environment setup:

```bash
bash scripts/run_smoke.sh
bash scripts/run_formal.sh full
```

Explain that the formal command runs/resumes all 140 jobs in one scope and finalizes the scope automatically.

- [ ] **Step 2: State the non-automatable boundary**

State that all six scopes cannot be chained on the current server because each scope must be transferred, verified, and manually cleaned before the next one starts. Keep transfer and cleanup instructions concise and explicit.

- [ ] **Step 3: Reclassify lifecycle subcommands as maintenance tools**

Keep `h100.sh check`, `status`, `scope`, and `finalize` in README only for diagnostics and recovery; do not present them as the normal run sequence.

- [ ] **Step 4: Reader-test the guides**

Verify that a reader can answer: which command uses one GPU, which command uses four, what `full` means, whether finalize is automatic, and why the six scopes cannot run back-to-back automatically.

### Task 4: Verify and publish

**Files:**
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: Add Bash syntax checks**

Add `bash -n scripts/run_smoke.sh scripts/run_formal.sh` to CI.

- [ ] **Step 2: Run all checks**

```bash
bash -n scripts/h100.sh scripts/run_smoke.sh scripts/run_formal.sh scripts/run_4gpu.sh
python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and the diff has no whitespace errors.

- [ ] **Step 3: Commit and push**

```bash
git add scripts/run_smoke.sh scripts/run_formal.sh tests/test_operator.py README.md RUN_TASK.md .github/workflows/tests.yml docs/superpowers/plans/2026-10-14-two-command-h100-entrypoints.md
git commit -m "Simplify H100 runs to two commands"
git fetch origin
git rebase origin/main
git push origin main
```

Expected: local and remote `main` resolve to the same commit.
