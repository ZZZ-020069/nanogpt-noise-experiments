# H100 Operator Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one safe shell entrypoint that hides the confirmed server paths and reduces input checking, smoke testing, formal scope launch, status inspection, and scope finalization to short subcommands.

**Architecture:** `scripts/h100.sh` is an operational wrapper around the existing tested modules and `scripts/run_4gpu.sh`; it does not duplicate training or scheduling logic. It fixes the server defaults to `/data/260010189/inputs` and `/data/260010171/results/nanogpt-v1.5`, permits environment overrides, records the smoke-test code commit, and refuses formal launch when the smoke test or code identity is inconsistent.

**Tech Stack:** Bash, Python 3.11, existing `nanogpt_noise` CLI modules, `unittest`.

---

### Task 1: Specify the operator command contract with tests

**Files:**
- Create: `tests/test_operator.py`
- Test: `tests/test_operator.py`

- [ ] **Step 1: Write failing integration tests**

Create a Bash-backed `unittest` fixture with temporary input/result directories and stub `git`/`python` executables. Test these exact behaviors:

```python
result = invoke('check', visible='0')
self.assertEqual(result.returncode, 0, result.stderr)
self.assertIn('nanogpt_noise.data', result.stdout)
self.assertIn('unittest discover -s tests -v', result.stdout)
self.assertIn('/openwebtext_subset', result.stdout)
self.assertIn('/frozen-artifacts', result.stdout)

smoke = invoke('smoke', visible='0')
self.assertEqual(smoke.returncode, 0, smoke.stderr)
self.assertTrue((run_base/'smoke/1gpu/full-muon-node10/status.json').is_file())
self.assertEqual((run_base/'smoke/operator_commit.txt').read_text().strip(), 'test-commit')

formal = invoke('scope', 'full', visible='0,1,2,3')
self.assertEqual(formal.returncode, 0, formal.stderr)
self.assertIn('nanogpt_noise.pool', formal.stdout)
self.assertIn((run_base/'full').as_posix(), formal.stdout)

invalid = invoke('scope', 'bad', visible='0,1,2,3')
self.assertNotEqual(invalid.returncode, 0)
```

The Python stub must return the registered input ID for `nanogpt_noise.data`, create a completed eight-step `status.json` for `nanogpt_noise.train`, and print all arguments so dispatch can be asserted. The Git stub must return `test-commit` for `git rev-parse HEAD` and an empty result for `git status --porcelain`.

- [ ] **Step 2: Run the test and verify failure**

Run:

```bash
python -m unittest tests.test_operator -v
```

Expected: failure because `scripts/h100.sh` does not exist.

### Task 2: Implement the unified H100 operator wrapper

**Files:**
- Create: `scripts/h100.sh`
- Test: `tests/test_operator.py`

- [ ] **Step 1: Implement fixed defaults and command parsing**

The script must use:

```bash
INPUT_ROOT="${INPUT_ROOT:-/data/260010189/inputs}"
DATA_DIR="${DATA_DIR:-$INPUT_ROOT/openwebtext_subset}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$INPUT_ROOT/frozen-artifacts}"
RUN_BASE="${RUN_BASE:-/data/260010171/results/nanogpt-v1.5}"
PYTHON="${PYTHON:-python}"
```

Accepted commands are `check`, `smoke`, `scope SCOPE`, `finalize SCOPE`, `status [SCOPE]`, and `--help`. Valid scopes are exactly `full`, `last1`, `last2`, `last3`, `last6`, and `last12`.

- [ ] **Step 2: Implement validation helpers**

Require the six registered files, a clean Git worktree, and the registered input identity:

```bash
EXPECTED_ID='9cabd9c54c6d717354f8f6750715f9280c5502f0a85a9032806b9c7956cbd76e'
identity="$($PYTHON -m nanogpt_noise.data --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --verify)"
[[ "$identity" == "$EXPECTED_ID" ]]
```

Run `sha256sum -c SHA256SUMS` when the input repository manifest exists. Print `df -B1` and `du -sh`, but do not enforce an automatic disk threshold.

- [ ] **Step 3: Implement lifecycle commands**

`check` runs input verification and the full CPU suite. `smoke` runs `check`, calls `bash scripts/run_4gpu.sh smoke` with `RUN_DIR="$RUN_BASE"`, verifies `status=completed`, `mode=smoke`, `step=8`, and writes `$RUN_BASE/smoke/operator_commit.txt`. `scope SCOPE` revalidates input, requires the smoke commit to equal the current clean commit, sets `RUN_DIR="$RUN_BASE/$SCOPE"`, and delegates to `run_4gpu.sh scope`. `finalize SCOPE` delegates to `python -m nanogpt_noise.scope finalize`. `status` only reads files and disk usage.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
python -m unittest tests.test_operator -v
python -m unittest discover -s tests -v
```

Expected: all tests pass; the suite count increases beyond 41.

### Task 3: Replace the long operator sequence with the wrapper commands

**Files:**
- Modify: `RUN_TASK.md`
- Modify: `README.md`

- [ ] **Step 1: Update the confirmed server paths**

Use `/data/260010189/inputs/openwebtext_subset` and `/data/260010189/inputs/frozen-artifacts` for inputs, while retaining `/data/260010171/nanogpt-noise-experiments` for code and `/data/260010171/results/nanogpt-v1.5` for results.

- [ ] **Step 2: Make the short task guide wrapper-first**

After environment activation and `python -m pip install -e '.[plots]'`, reduce normal operation to:

```bash
bash scripts/h100.sh check
bash scripts/h100.sh smoke
bash scripts/h100.sh scope full
bash scripts/h100.sh finalize full
```

State clearly that `smoke` needs one visible H100 and `scope` needs exactly four visible H100s, so they are normally separate allocations.

- [ ] **Step 3: Retain manual commands as troubleshooting detail in README**

Document every wrapper subcommand, environment override, smoke status path, and the fact that the wrapper does not activate Conda, install CUDA/PyTorch, pull code, transfer results, delete scope data, or automatically stop on low disk space.

- [ ] **Step 4: Validate documentation**

Run:

```bash
git diff --check
rg -n '/data/260010171/inputs|run_4gpu.sh smoke|run_4gpu.sh scope' README.md RUN_TASK.md
```

Expected: the old input path is absent; direct low-level commands appear only in the troubleshooting/reference explanation, not as the primary operator path.

### Task 4: Commit and publish

**Files:**
- Add: `scripts/h100.sh`
- Add: `tests/test_operator.py`
- Add: `docs/superpowers/plans/2026-10-14-h100-operator-wrapper.md`
- Modify: `README.md`
- Modify: `RUN_TASK.md`

- [ ] **Step 1: Confirm the final diff and clean tests**

```bash
git diff --check
python -m unittest discover -s tests -v
git status --short
```

- [ ] **Step 2: Commit**

```bash
git add scripts/h100.sh tests/test_operator.py README.md RUN_TASK.md docs/superpowers/plans/2026-10-14-h100-operator-wrapper.md
git commit -m "Add unified H100 operator workflow"
```

- [ ] **Step 3: Rebase on the latest remote and push**

```bash
git fetch origin
git rebase origin/main
git push origin main
```

Expected: local and remote `main` resolve to the same commit.
