# LM Head-only Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Execution in this session is inline; the optional sub-skills are not installed. The user already requested implementation, so no separate execution approval is needed.

**Goal:** Implement the published LH-QP design with isolated G/T/K runners, Hessian measurements and executable Linux launch scripts, without changing legacy last-k behavior.

**Architecture:** Add nanogpt_noise/lm_head as a separate package. Reuse immutable data, budgets, LR proposals, numerical checks and atomic storage; implement Head-only model/optimizer/checkpoint semantics separately. Existing production entry points stay unchanged.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, Bash; CPU tests locally and explicit CUDA smoke on the target host.

The experimental authority remains the workspace main design chapter 14; this file only tracks implementation.

## Tasks

- [x] Add model.py and protocol.py: same-value untie, complete Head scope, Muon-head, G/T/K losses, isolated code/input identity. Check initial logits, frozen weights, gradients and surrogate derivatives against autograd on small matrices.
- [x] Add train.py: retain the old numerical/budget sequence, evaluate real CE for all objectives, save only Head plus optimizer state and source identity, retain analysis anchors, reject incompatible resume. Compare interrupted and continuous trajectories for all optimizers and objectives.
- [x] Add curvature.py: deterministic windows/parameter pools, FP64 analytic blocks, inclusion-probability spectral moments, trace, K fitting/held-out validation, stage/checkpoint manifests. Check rectangular blocks and weighted moments against exact small Hessians.
- [x] Add runner.py and __main__.py: clean/grid jobs, sequential eight-candidate LR search, dynamic GPU workers, pause/resume, safe output namespaces, read-only dry run, summary export. Test missing/changed inputs, paused jobs and deterministic replay.
- [x] Add scripts/run_lm_head.sh and LM_HEAD_RUN.md: check, smoke, clean, grid, pause, resume, curvature and K commands with explicit input/output arguments. Check Bash syntax, CLI help and dry-run behavior.
- [x] Run meaningful new CPU numerical/resume/end-to-end tests and existing baseline regression tests; record results and unresolved target-hardware checks.
- [x] Commit and publish implementation on the isolated design branch. Verify old source/entry points are unchanged and remote content matches.

## Validation commands

```bash
python -m unittest tests.test_lm_head -v
python -m unittest discover -s tests -v
bash -n scripts/run_lm_head.sh
python -m nanogpt_noise.lm_head --help
```

No formal training or server deployment is part of this implementation turn.
