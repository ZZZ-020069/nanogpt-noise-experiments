# Scope-isolated execution and storage-retention plan

> **For agentic workers:** Execute every checkbox in order. Keep `nanogpt_experiment_design.md` and its repository snapshot byte-identical after design edits.

**Goal:** Make each NanoGPT scope an independently runnable, transferable, verifiable, and removable batch whose estimated four-GPU peak stays below 200 GB.

**Architecture:** A scope run owns one output directory and exactly 140 global jobs. Candidate runs remain resumable while active; after each terminal candidate its step records are compacted losslessly, and after the search record is durable only the current winner retains one full checkpoint. A scope manifest closes the batch and permits a downloaded copy to be verified before server-side removal.

**Tech stack:** Python 3, PyTorch, Bash, gzip JSONL, SHA-256 manifests, `unittest`.

---

- [x] Add failing tests for lossless/idempotent step compaction and winner-only checkpoint retention.
- [x] Add failing tests for scope bounds, single-scope directory ownership, completion manifests, copied-data verification, and tamper detection.
- [x] Add failing launcher and analysis tests for named-scope execution and multiple downloaded scope roots.
- [x] Implement candidate compaction and checkpoint retention, then update training to avoid persistent `best.pt` and duplicate terminal checkpoint writes.
- [x] Implement scope preparation, size estimates, finalization, verification, and guarded cleanup commands.
- [x] Add named `scope` launcher mode and make the Slurm template scope-based while keeping custom ranges as an expert interface.
- [x] Let analysis read the six downloaded scope directories without merging them.
- [x] Rewrite the README as an operator runbook: run, resume, finalize, transfer, verify, clear, and advance to the next scope.
- [x] Update the authoritative design and repository snapshot to v1.5, including retention semantics and the under-200-GB peak estimate.
- [x] Run the complete CPU test suite, design snapshot check, shell syntax checks, and repository consistency checks.
- [x] Commit and push the verified changes to `origin/main`.
