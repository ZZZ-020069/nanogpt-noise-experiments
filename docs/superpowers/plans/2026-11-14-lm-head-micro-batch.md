# LM Head Micro Batch Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Execution here is inline; the optional sub-skills are unavailable and the user already authorized implementation.

**Goal:** Adopt micro=16 as the configurable default and reject mixed-micro paired experiments, retaining the global batch and numerical operators.

**Architecture:** Keep changes inside the LM Head package and its launcher. Add micro to the root identity and verify every reported candidate against it. Keep the immutable v1.8 code and existing published design snapshot unchanged.

**Tech Stack:** Python, PyTorch, Bash, unittest; Windows CPU tests and Git Bash stubs locally.

---

- [x] Register the guide's accepted decisions in the sole main design, section 14.17: default 16, candidates 16/32, one selected micro for G/T/K, no checkpoint migration, GPU selection still pending.
- [x] Add tests in tests/test_lm_head_micro.py for default propagation, root/CLI/report rejection, missing old identity fields, changed-micro resume, actual 200-sequence evaluation tails, and 16 full diagnostic batches.
- [x] Set DEFAULT_MICRO = 16 in nanogpt_noise/lm_head/protocol.py; use it in runner.py, train.py, measure.py and __main__.py. Set the Bash default to 16. Keep Budget.accumulations(1, micro), loss/accumulation, and one noise injection per step unchanged.
- [x] Bind micro in lm_head_run.json; use require_run_micro before direct smoke, queue and run_node calls; compare execution.micro with the root in analyze.report before emitting reports. Update tiny test calls to explicit micro=2 or 4.
- [x] Verify 512 // 16 == 32 and 512 // 32 == 16, fixed 524288 tokens/step and 3815 steps. Evaluate 200 sequences against an independent per-sequence mean at micro 16/32; compare offline feature, gradient and curvature diagnostics including partial chunks.
- [x] Document temporary per-micro smoke roots, G/T/K resource checks, K's approved-artifact prerequisite, final common micro and a fresh formal root. Keep T/K operators unchanged.
- [x] Run the full unittest discovery with Git Bash first on PATH, plus Bash syntax and documentation audits; record passed/skipped counts and GPU limitations.
- [x] Refresh implementation provenance, commit and push only design/lm-head-only-20261114; verify main and the historical design snapshot are unchanged.

Validation:

~~~bash
python -m unittest tests.test_lm_head_micro -v
python -m unittest discover -s tests -v
bash -n scripts/run_lm_head.sh
~~~

No server access, GPU smoke, formal training, existing-result cleanup or live-code updates are authorized by this change.
