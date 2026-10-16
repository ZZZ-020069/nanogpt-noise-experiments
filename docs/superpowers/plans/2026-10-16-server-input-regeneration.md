# Server-side Input Regeneration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the H100 operator generate, freeze, register, and verify one shared OpenWebText input bundle directly on the server without SCP or Git LFS.

**Architecture:** Keep corpus preparation in `nanogpt_noise.prepare`, add an immutable registration layer that binds the generated corpus to initialization and evaluation-window artifacts, and teach the launch wrapper to accept that registered identity. The generated bundle is created once under `/data/260010171/inputs` and reused by every scope; a rerun verifies it instead of regenerating it.

**Tech Stack:** Python 3.11, PyTorch 2.8, NumPy, Hugging Face Datasets streaming, tiktoken, Bash, unittest.

---

### Task 1: Register a generated input bundle

**Files:**
- Create: `nanogpt_noise/input_bundle.py`
- Test: `tests/test_prepare.py`

- [x] Add tests that build a tiny corpus/artifact fixture, register it, verify its identity, and reject corpus tampering or a changed registration marker.
- [x] Implement `register_bundle(input_root, data, artifacts)` to call the existing cryptographic verifier, atomically write `registration.json` and `SHA256SUMS`, and refuse to replace a different registration.
- [x] Implement `verify_bundle(input_root, data, artifacts)` to require the recorded identity to equal the recomputed identity.
- [x] Run `python -m unittest tests.test_prepare -v` and require all tests to pass.

### Task 2: Make corpus generation restartable and bundle-aware

**Files:**
- Modify: `nanogpt_noise/prepare.py`
- Test: `tests/test_prepare.py`

- [x] Test the token fill boundary behavior with a fake stream and encoder, plus the completed-bundle reuse path.
- [x] Preserve the fixed source revision, split, shuffle seed, buffer, GPT-2 tokenizer, 10M validation tokens and 1B training tokens.
- [x] Add `--input-root` to create `openwebtext_subset`, freeze `frozen-artifacts`, and register them; add `--restart-incomplete` that removes only the two named `.incomplete` files after the preparation lock is acquired.
- [x] If a registered bundle exists, verify and print its identity without downloading or overwriting anything.
- [x] Run focused tests.

### Task 3: Add a one-command ACP input preparation mode

**Files:**
- Create: `requirements-prepare.txt`
- Modify: `scripts/run_acp.sh`
- Modify: `scripts/h100.sh`
- Modify: `tests/test_acp.py`
- Modify: `tests/test_operator.py`

- [x] Add `bash scripts/run_acp.sh inputs`, which reuses the isolated runtime, installs pinned preparation dependencies, and invokes bundle preparation at `/data/260010171/inputs`.
- [x] Change launch verification to accept `registration.json` when present while retaining the legacy fixed identity fallback for the already published bundle.
- [x] Test routing with stubs and test both registered and legacy identities.
- [x] Run Bash syntax checks and the full unittest suite.

### Task 4: Record the protocol change and operator procedure

**Files:**
- Modify: `VSCode下载/New/nanogpt_experiment_design.md`
- Modify: `docs/nanogpt_experiment_design.md`
- Modify: `README.md`
- Modify: `RUN_TASK.md`
- Modify: `nanogpt_noise/protocol.py`

- [x] Bump the protocol to v1.6 and state that one freshly generated input realization is frozen before the first smoke and reused unchanged for all six scopes.
- [x] Explain that its exact losses are not numerically interchangeable with the earlier registered realization, while within-realization optimizer/noise comparisons remain paired and fair.
- [x] Document the single input command, expected time/network/storage behavior, registration files, verification, interruption handling, and the prohibition on regeneration between scopes.
- [x] Keep the main design and repository snapshot byte-identical.
- [x] Run design consistency, documentation path, and full test checks.

### Task 5: Commit and publish

**Files:** all files above.

- [x] Confirm `git diff --check`, a clean generated-file boundary, and no input binaries are staged.
- [x] Commit the implementation with a focused message.
- [x] Push `main` to `origin` and report the commit plus the exact operator command.
