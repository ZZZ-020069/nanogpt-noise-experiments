# Git Safe Directory Startup Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both public H100 launch scripts work when the platform runtime user differs from the repository owner, without requiring a separate manual Git configuration command.

**Architecture:** Configure Git's `safe.directory` as process-scoped environment configuration inside `scripts/h100.sh`, using the script-derived absolute repository root. Export the configuration before every Git or Python child process so both the wrapper's `git status` and Python metadata collection's `git rev-parse` accept exactly this repository. Do not change global Git configuration or trust a wildcard path.

**Tech Stack:** Bash, Git environment configuration, Python `unittest`.

---

### Task 1: Reproduce the platform ownership failure in tests

**Files:**
- Modify: `tests/test_operator.py`

- [ ] **Step 1: Extend the Git stub**

When `REQUIRE_SAFE_DIRECTORY=1`, make the Git stub return the platform-style `detected dubious ownership` failure unless one exported `GIT_CONFIG_KEY_n=safe.directory` has the matching `GIT_CONFIG_VALUE_n` repository path.

- [ ] **Step 2: Add a public-entrypoint test**

Invoke `scripts/run_smoke.sh` with safe-directory enforcement enabled. Assert that it succeeds and that the exact repository root, rather than `*`, reaches the Git stub.

- [ ] **Step 3: Run the focused test and verify failure**

```bash
python -m unittest tests.test_operator -v
```

Expected: the new test fails with the simulated dubious-ownership error.

### Task 2: Inject exact process-scoped Git trust

**Files:**
- Modify: `scripts/h100.sh`
- Test: `tests/test_operator.py`

- [ ] **Step 1: Add a Git environment helper**

Read the existing `GIT_CONFIG_COUNT`, append one configuration entry with key `safe.directory` and value `$REPO_ROOT`, export all three variables, and call the helper before command dispatch.

- [ ] **Step 2: Preserve existing Git environment entries**

Use the next numeric index rather than resetting `GIT_CONFIG_COUNT`, so platform-provided Git configuration is not discarded.

- [ ] **Step 3: Run focused tests**

```bash
python -m unittest tests.test_operator -v
```

Expected: all operator tests pass, including the ownership regression.

### Task 3: Update operator instructions

**Files:**
- Modify: `RUN_TASK.md`
- Modify: `README.md`

- [ ] **Step 1: Document automatic handling**

State that `run_smoke.sh` and `run_formal.sh` automatically handle the platform's repository-owner mismatch for the exact project path; no separate `git config --global --add safe.directory ...` command is needed.

- [ ] **Step 2: Add troubleshooting guidance**

Explain that a remaining dubious-ownership error likely means an old commit or a launch path bypassing the public scripts; instruct the runner to pull the latest commit and use the public entrypoints.

- [ ] **Step 3: Keep the normal instructions to two commands**

Do not add another startup script or another required step.

### Task 4: Verify and publish

**Files:**
- Modify: `.github/workflows/tests.yml` only if command coverage changes.

- [ ] **Step 1: Run validation**

```bash
bash -n scripts/h100.sh scripts/run_smoke.sh scripts/run_formal.sh
python -m unittest discover -s tests -v
git diff --check
```

Expected: all tests pass and no whitespace errors are reported.

- [ ] **Step 2: Commit and push**

```bash
git add scripts/h100.sh tests/test_operator.py README.md RUN_TASK.md docs/superpowers/plans/2026-10-15-git-safe-directory.md
git commit -m "Handle platform Git ownership automatically"
git fetch origin
git rebase origin/main
git push origin main
```

Expected: local and remote `main` resolve to the same commit.
