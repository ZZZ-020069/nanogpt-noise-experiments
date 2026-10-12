# Four independent H100 workers implementation plan

> **Protocol boundary:** This changes orchestration only. One job remains one
> `scope / optimizer / noise node`, and its eight learning-rate candidates remain
> sequential because candidates 6–8 depend on earlier outcomes.

## Goal

Validate that one H100 can carry one candidate, then use four H100s as four
independent workers. Each worker claims one job at a time and runs that job's
eight candidates in order; when it finishes, it claims the next unstarted job.

## Implementation

1. Update `nanogpt_noise/preflight.py` so visible-device validation and training
   world size are separate. The production pool checks four visible H100-class
   devices while computing accumulation for a one-GPU training world
   (`128` at micro-batch `4`). NCCL is required only for an actual multi-rank
   training world.
2. Add `nanogpt_noise/pool.py`. It accepts an inclusive job-index interval and
   four exact `CUDA_VISIBLE_DEVICES` tokens, starts one `nanogpt_noise.node`
   subprocess per GPU, and dynamically assigns the next job to whichever worker
   becomes free. Each job gets an append-only launcher log. The first child
   failure stops new assignments and terminates active children, so rerunning the
   same interval resumes the interrupted candidate rather than silently replacing
   it.
3. Replace the DDP behavior of `scripts/run_4gpu.sh`:
   - `smoke` isolates one selected GPU and runs eight diagnostic steps;
   - `node INDEX` isolates one selected GPU and runs one complete eight-candidate
     job;
   - `queue START END` validates four visible GPUs and starts the dynamic pool.
   No production mode invokes `torch.distributed.run`.
4. Update launcher/preflight tests first, add pool scheduling and fail-fast tests,
   then implement until the focused and full CPU suites pass.
5. Rewrite `README.md` around the actual operator sequence: prepare data, validate
   one H100, run a one-GPU smoke test, run jobs `0..3` as the first four-card
   acceptance, then submit larger resumable intervals. Explicitly distinguish
   formal results from smoke tests and CPU/DDP protocol checks.

## Verification

Run:

```bash
python -m unittest tests.test_launcher tests.test_pool -v
python -m unittest discover -v
bash -n scripts/run_4gpu.sh
```

The local/CI checks validate orchestration and protocol behavior only. The final
single-H100 capacity check must be performed on the senior student's H100 server.
