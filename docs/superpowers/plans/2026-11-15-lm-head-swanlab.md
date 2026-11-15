# LM Head SwanLab Implementation Plan

**Goal:** Enable observable H100 Head-only runs with isolated cloud uploads.
**Execution:** Inline under the user's implementation request.
**Architecture:** Keep the frozen last-k telemetry implementation unchanged. A small Head adapter validates environment settings and adds objective evaluation/state metrics. H100 enables it by default; cloud errors remain isolated in the existing process.
**Tech stack:** Python, unittest, Bash, SwanLab SDK.

- [x] Add tests in tests/test_lm_head_telemetry.py for startup, resume, disabled/test mode, secret-free preflight, payload and upload failures. Extend tests/test_lm_head_h100.py for enabled defaults and missing-key rejection.
- [x] Create nanogpt_noise/lm_head/telemetry.py with check_environment() and a Telemetry subclass. Require a nonempty key and importable SDK only when enabled; never print credentials. Extend scalar metrics via super().log().
- [x] In nanogpt_noise/lm_head/train.py import the adapter and replace the try-block telemetry=None with Telemetry(output, cloud_config, identity, mode, step). Include actual input/code identity, micro and target budget in cloud_config; keep local candidate config unchanged.
- [x] In scripts/run_lm_head_h100.sh export default NANOGPT_SWANLAB=1 and SWANLAB_PROJ_NAME=nanogpt-lm-head; preflight check/smoke/clean/grid/resume. Preserve plan/status/report and curvature without SDK checks. Add settings to scripts/lm_head_h100.env.example.
- [x] Update LM_HEAD_RUN.md, README.md and implementation records for SDK installation, environment-only Key, dashboard metrics, fresh v3 paths and actual smoke cloud acceptance. No GPU execution.
- [x] Run `python -m unittest tests.test_telemetry tests.test_lm_head_telemetry tests.test_lm_head tests.test_lm_head_micro tests.test_lm_head_shell tests.test_lm_head_h100 -v`, Bash syntax and command previews. Refresh the implementation manifest; commit and publish the fresh v3 tag without moving older releases.

Validation: 45 tests passed in 59.898 seconds. New tests first failed on the missing adapter, then passed after wiring it. The initial unprivileged Bash checks hit the Windows MSYS sandbox restriction; the approved Bash environment passed the combined suite. No cloud credentials or GPU jobs were used.
