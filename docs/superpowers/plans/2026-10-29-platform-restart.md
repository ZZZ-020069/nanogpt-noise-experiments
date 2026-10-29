# Platform stop and restart implementation plan

**Goal:** Restart the same persistent-output training command after a batch platform stops or kills the previous container, without a mid-run interactive command.

**Architecture:** POSIX directory locks use a stable `.lease` inode and nonblocking `flock`; `.lock` is an atomic owner marker, recoverable only with the new lock format and the exclusive kernel lease. Never unlink `.lease` or automatically adopt legacy locks. The formal Bash supervisor forwards TERM/INT and waits for the pool; ACP execs this supervisor. The same command resumes durable candidates and closes a finalized scope idempotently.

**Tech stack:** Bash, Python stdlib `fcntl`, PyTorch tiny CPU fixtures, Linux subprocess integration tests.

1. Add lock tests for live-owner rejection, forced-kill restart, changed container hostname, malformed/legacy owner rejection, and persistent lease inode identity. Expected: exclusive lock acquisition fails while another process lives and succeeds after its SIGKILL.
2. Add a `run_child` Bash helper that records stop requests, forwards TERM/INT to its child, retries interrupted wait, and returns 75 after a requested stop. Use it for both scope execution and finalization; ACP's formal branch uses exec.
3. Add Linux integration tests launching the real formal supervisor plus a tiny CPU trainer. TERM must save a positive-step checkpoint and skip finalization. SIGKILL of the owned process group must release kernel locks; restarting with the same output must resume identical model and optimizer states. Concurrent restart must refuse the active lease.
4. Keep fresh-input version v1.8, budget, search, checkpoint interval and resume identity checks. Remove only recognized orphan atomic files while holding the corresponding writer lease; exclude persistent lease files from scope manifests.
5. Update authoritative design and snapshot, RUN_TASK/README for a single platform startup command using exec, platform stop, and restart with the same command. Keep manual pause as an optional separate control path. State that the external run_v18_ready.sh and actual platform grace period remain unverified.
6. Run CPU suite, Bash syntax, design synchronization and Linux CI. Commit and push. No GPU training or real input preparation.

Validation: `python -m unittest tests.test_platform_restart -v`; `python -m unittest discover -s tests -v`; `python scripts/check_design.py`; `for script in scripts/*.sh; do bash -n "$script"; done`.
