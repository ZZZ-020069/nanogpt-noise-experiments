> Historical implementation plan. Current protocol and launch order are v1.7 in the design snapshot and RUN_TASK.md; earlier budget choices below are not current instructions.

# Joint prefix pilot implementation plan

> **For agentic workers:** Execute the following tasks inline in the current session. The user explicitly authorized implementation and will run the GPU workload. No remote training, commit, push, environment installation, or new chat is part of this task.

**Goal:** Deliver the approved 24-candidate joint screening pilot with two independent GPU workers, strict same-candidate recovery, and descriptive early-prefix comparisons.

**Architecture:** Add a self-contained Python supervisor/worker script and Bash entrypoint. Reuse `train.run(..., smoke_steps=573)` and immutable existing inputs; mark each candidate config with the pilot policy and manifest identity. Keep the production trainer and research constants unchanged. Store the scientific specification only in the master design, section 12.6.

**Tech Stack:** Python standard library, existing PyTorch/NumPy trainer, unittest, Bash.

## Files and responsibilities

- `scripts/joint_pilot.py`: fixed candidate plan, immutable execution manifest, subprocess lifecycle, checkpoint recovery, lossless compaction, summaries.
- `scripts/run_joint_pilot.sh`: paths/environment forwarding; fresh output by default, explicit directory for resume.
- `tests/test_joint_pilot.py`: plan coverage, identity rejection, actual tiny-model recovery, failure distinction, isolated worker scheduling, truthful summaries.
- `docs/JOINT_PILOT.md`: operator commands, completion/recovery, storage and interpretation.
- Main design `../nanogpt_experiment_design.md`: user-approved pilot scope and implementation validation record; formal v1.6 parameters remain unchanged.

## Tasks

- [x] Freeze and test the candidate plan: 24 unique candidates; five clean LR exponents per optimizer and one small-noise center; 573 steps, original schedule. Dry-run must need neither CUDA nor inputs and must not create output.

```python
specs = pilot.candidates()
assert len(specs) == len({s['name'] for s in specs}) == 24
assert sum(s['node'] == 0 for s in specs) == 20
assert sum(s['node'] == 1 and s['exponent'] == 0 for s in specs) == 4
```

- [x] Implement the manifest and worker adapter. Bind input ID, package hash, runner hash, runtime, micro-batch, candidates and prefix length. Reuse the existing schedule; reject changed identity on recovery. Retain all candidate checkpoints; compact terminal logs without removing scientific fields. Test interruption and exact tiny-model resumed/uninterrupted equality.
- [x] Implement the two-GPU supervisor. One child per assigned GPU token, fresh process per candidate, no DDP, fixed plan order. Continue numerical failures; stop assignment on infrastructure failure or signal, terminate active children gracefully, return nonzero, recover only into the same directory. Test with injected subprocesses and synthetic inputs.
- [x] Write `summary.json`, `candidates.csv`, `evaluations.csv`, `report.txt`. Compare five versus three **initial** LR results only; report missing points and failed candidates explicitly. Compare clean/noisy center losses only at shared token counts. Never label this as complete 3+2 versus 5+3 validation or 2B adequacy.
- [x] Add Bash entrypoint and operator documentation. Package the entrypoint, runner and docs in a ZIP with SHA-256 so the user can transfer them to the existing remote checkout without updating the production code.
- [x] Run focused tests, the repository CPU suite, Python compilation, Bash syntax where available, and read-only dry-run. Record actual results and hardware limitations in the master design; provide the concrete remote command.

## Validation commands

```powershell
& 'D:\Anaconda\python.exe' -m unittest discover -s tests -p test_joint_pilot.py -v
& 'D:\Anaconda\python.exe' -m unittest discover -s tests -v
& 'D:\Anaconda\python.exe' scripts/joint_pilot.py --data unused --artifacts unused --output unused --dry-run
```

Passing CPU tests is implementation evidence only. This session does not run the 124M GPU pilot.

Result: 80 repository tests passed, including 16 new pilot tests; real tiny-model coverage of all 24 candidates and all-optimizer bitwise recovery. Bash syntax and actual Bash dry-run passed. Windows sandbox MSYS initialization required running the existing shell tests outside the sandbox with temporary fixtures still inside this workspace. No GPU execution, commit, or push occurred.
