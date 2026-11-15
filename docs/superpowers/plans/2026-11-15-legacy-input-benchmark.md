# Legacy-input hardware benchmark

User authorized old data for throughput only. Implement inline in the isolated branch.

- Add a bounded benchmark mode to Head run: full model, verified source hashes, exactly 32 steps, current Budget, no formal passed proof. Keep source input identity unchanged.
- Add empirical full-size K statistics with a distinct benchmark schema and passed=false. Formal load_statistics must reject it. Reuse the same K math and full vocabulary.
- Add explicit --benchmark-only to the selector; separate worker command, proof filename and mode. Record both code scripts and input provenance. H100 accepts the chosen micro as a hardware suggestion but validates its own formal inputs/K before smoke.
- Test formal rejection, 32-step bounds, source protocol checks, benchmark proof separation and handoff differences. Run existing Head regression for training changes.
- Publish a new fixed version, deploy separately, recheck GPU status, compute real empirical K statistics, and run both 12-candidate lanes. Preserve previous deployment and inputs. Inspect the actual outputs through completion and report any failures honestly.

Files: nanogpt_noise/lm_head/benchmark.py, train.py, scripts/benchmark_lm_head.py, select_lm_head_micro.py, tests/test_lm_head_benchmark.py, LM_HEAD_RUN.md and implementation records. Main design section 14.19 is the authority for this scoped exception.

Completed: version 74af8db deployed; 39 distinct local checks and 14 remote CPU-only checks passed; 24 real GPU benchmark candidates passed in 1198.019 seconds. Selection=32, ~0.99% lower summed candidate wall time. Evidence and independent frozen-parameter audit saved. No formal training.
