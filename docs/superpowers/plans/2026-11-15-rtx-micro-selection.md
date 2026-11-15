# RTX micro selection implementation plan

> Execution: implement inline in the existing isolated checkout. The user has authorized implementation and RTX selection; no additional execution-choice approval is needed.

**Goal:** Run micro 16 and 32 on separate RTX GPUs, preserve every attempt, and produce a complete G/T/K selection report for subsequent H100 smoke.

**Architecture:** A Python supervisor in scripts launches the existing four-optimizer smoke CLI. Each micro has a serial G/T/K lane; two lanes run concurrently. Training code and mathematics stay frozen. Selection identity additionally hashes the supervisor and records GPUs, software, input and K artifact.

**Tech Stack:** Python standard library, existing PyTorch training CLI, Bash, nvidia-smi.

### 1. Establish prerequisites

- [x] Read main design and inspect current smoke implementation.
- [x] Read-only SSH GPU/process/disk check; two GPUs idle at 11:13 CST.
- [x] Inspect input: RTX has v1.4/4960-step artifacts, no registered v1.8 bundle or approved K. Request the current input location; do not relabel old inputs.
- [x] Record current runtime decision in main design section 14.18.

### 2. Build and verify supervisor

Files: scripts/select_lm_head_micro.py, scripts/select_lm_head_micro.sh, tests/test_lm_head_selection.py.

- [x] Add pure report validation tests: incomplete K cannot select; OOM can disqualify a micro, infrastructure failure cannot; partial resumed timings cannot count as full 32-step timings; mismatched proof rejected; compare only complete four-optimizer results.
- [x] Implement GPU inventory and busy-card rejection, immutable identity, parent lease, two serial subprocess lanes, stop signal propagation, and append-only attempt directories. Failed attempts are retried only under explicit --retry-failed; completed attempts are reused only after artifact revalidation.
- [x] Report objective and optimizer timings/peak allocation. Select from eligible micros by the sum of 12 full smoke wall times; exact ties choose 16. Require at least 10% allocation headroom as a runtime planning rule, not a claim about free device memory. H100 must validate its own memory.
- [x] Keep missing-K stage explicitly incomplete. Reuse G/T after a genuine approved K artifact is supplied; lock the K hash on first use.
- [x] Run: python -m unittest tests.test_lm_head_selection -v, then existing LM Head Python and shell regression tests. Expected: all pass; no GPU benchmarks inferred from CPU fixtures.

### 3. Deliver and execute when prerequisites exist

Files: LM_HEAD_RUN.md, README.md, docs/experiments/lm-head-only/RTX_SELECTION_RECORD.md and rtx-selection-manifest.json.

- [x] Replace H100 dual-micro selection instructions with RTX selection, followed by only selected-micro H100 acceptance. Include exact GT, curvature, all/resume, report and handoff commands.
- [x] Save real GPU evidence and input blocker; distinguish hardware inspection from a completed selection.
- [x] Commit/push the isolated branch; deploy to a fresh remote directory, never the old running checkout.
- [x] Verify remote help/preflight and ten CPU-only Linux tests. Old input is rejected before GPU execution.
- [ ] Run real selection after the current registered v1.8 input is provided; K still requires genuine approval. This step is blocked on missing input, not marked complete.

Core report rule:

~~~python
complete = all(group['status'] in ('passed', 'oom') for group in groups)
eligible = [micro for micro in (16, 32)
            if all(g['status'] == 'passed' and g['allocation_headroom_ok']
                   for g in groups if g['micro'] == micro)]
selected = min(eligible, key=lambda m: (sum(g['wall_seconds'] for g in groups
                    if g['micro'] == m), m)) if complete and eligible else None
~~~

Review: six objective groups must exist explicitly; missing groups are pending. An old successful report must never survive unvalidated changed evidence. No formal-training command is dispatched by this supervisor.

Execution checkpoint: implementation and 35 distinct checks passed. Deployment verification passed at fixed implementation commit 2e8c723. Real selection remains blocked on the current registered input; no candidate has run.
