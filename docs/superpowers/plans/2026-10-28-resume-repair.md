# Resume repair implementation plan

**Goal:** Preserve every recoverable candidate and final winner across search restarts; provide cooperative pause/resume for the four-worker scope.

**Architecture:** Retention receives the explicit terminal-candidate set already replayed, never scans ahead for deletion. Replay reconstructs search decisions without erasing later candidates. A scope pause-request file is checked before candidate dispatch and after complete optimizer steps. A pool lock prevents concurrent supervisors; explicit resume clears the request only while owning that lock.

**Tech stack:** Python, PyTorch CPU fixtures, Bash operator wrappers, existing JSON and atomic checkpoint utilities.

- [ ] Reproduce in `tests/test_resume.py`: pause candidate 1 after candidate 0 completed; assert its saved step is loaded and all final tensors/search decisions match uninterrupted execution. Repeat a completed search and assert winner bytes survive. Exercise all four optimizers.
- [ ] Change `storage.retain_winner_checkpoint(job, winner, *, completed)` to delete only indexed completed candidates; node passes `[record['candidate'] for record in records]`. Missing required final winner is an error. Keep checkpoint budget unchanged.
- [ ] Trim replayed per-step diagnostics above the loaded checkpoint; test a hard interruption with stale later diagnostics. Reject a missing checkpoint if status proves positive training progress.
- [ ] Add `pause_file` to train/run and node propagation; check at step boundaries and before starting a candidate. Preserve complete optimizer states and reconstruct noise/data from step.
- [ ] Add pool-level lock, external stop checks, dispatch race protection, and `--resume-paused`. Pause requests remain until explicit resume.
- [ ] Add `scripts/pause.sh`, ACP `pause` and `resume` routes, and scope-aware operator instructions. Pause must avoid package installation, wait for pool lock removal, and never use SIGKILL or remove stale locks automatically.
- [ ] Run focused failing tests before fixes, then CPU suite, shell syntax, design synchronization and Linux CI. Record CPU versus H100 validation limits. Commit and push without starting real training.

Commands: `python -m unittest tests.test_resume -v`; `python -m unittest discover -s tests -v`; `python scripts/check_design.py`; `bash -n scripts/pause.sh`.

Local implementation and validation complete: 105 tests passed. Remote publication/CI are recorded by the release commit and PROJECT_STATE, not assumed from local tests.
