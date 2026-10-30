# Dynamic row-priority scheduling Implementation Plan

**Goal:** Remove row barriers while preserving coordinate priority, stable IDs and automatic reports.
**Architecture:** Flatten row batches into one existing dynamic pool. Track successful row completion under a short lock. Wake a background reporter; never publish reports in GPU workers.
**Tech Stack:** Python threading/queue, unittest, existing Bash supervisor.

- [x] Replace per-row pool invocation in nanogpt_noise/pool.py with one pool and tracked runner. Retain failure and pause checks.
- [x] Add nonblocking reporter notification in nanogpt_noise/progress.py; remove misleading idle-GPU estimates because rows overlap.
- [x] Test a blocked first-row job while later rows finish; assert every ID once, same-GPU reuse, row callbacks once, stop/failure behavior and background notification.
- [x] Update authoritative design, README, snapshot and validation evidence. Run full CPU tests and Bash checks, push and check Linux CI. No GPU training.
