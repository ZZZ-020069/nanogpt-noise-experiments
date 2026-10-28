# Two-GPU short timing test

This is a performance smoke test, not a convergence pilot or formal experiment.
It uses the current v1.7 124M trainer, last1 scope by default, micro-batch 4, global batch 512
sequences, BF16, no clipping, and node 10 (sigma1=0.03, sigma2=0.01).
Two independent workers run AdamW/Muon, then Adam/SGD, for 20 steps each.
The current 2B LR schedule is retained; short smoke results do not establish
late-training numerical stability or optimizer quality.

## Prerequisites

- Linux machine with two allocated RTX PRO 6000 Blackwell GPUs visible to CUDA.
- Existing project environment (Python >=3.11 and compatible CUDA PyTorch;
  the project environment uses PyTorch 2.8/CUDA 12.8). Environment installation
  and input preparation are outside the timing deadline.
- Existing registered openwebtext_subset and frozen-artifacts inputs. Copy the
  same registered bundle if moving machines; do not regenerate a different
  corpus for the timing comparison.
- About 10 GB writable space for four temporary smoke checkpoints/logs.

## Run

Put benchmark_time.py and benchmark_time.sh in this repository's scripts folder.
Activate the existing CUDA Python environment, enter the repository, and run:

```bash
INPUT_ROOT=/data/260010171/inputs-v1.7 bash scripts/benchmark_time.sh
```

If inputs live elsewhere, replace INPUT_ROOT with their parent directory. Set
PYTHON=/absolute/path/to/python if necessary. Respect the scheduler's existing
CUDA_VISIBLE_DEVICES; if none is set, the script assumes allocated GPUs 0 and 1.
DATA_DIR and ARTIFACT_DIR may override the individual input directories.
BENCH_OUT can name a new, unused output directory on a writable disk.

The training subprocess deadline is 30 minutes, plus up to about 20 seconds
for graceful process termination. The script does not install dependencies,
download data or start a formal run. It retains incomplete logs on timeout.
Do not restart into the same output directory. A new invocation uses a new
timestamped directory. No checkpoints are automatically deleted.

## Outputs and interpretation

The final REPORT line points to summary.json. Return that file and the four
optimizer console.log files. Each successful worker also writes timing.json.
If a worker fails or times out, the report explicitly omits aggregate estimates.

The estimate excludes the first three training steps from steady-state timing,
includes regular diagnostic steps, measures validation and checkpoint saving,
and computes each optimizer's contribution separately. Reports include:

- Four representative candidates to 2B (measured scope).
- The current 35-node / 8-LR / 2B scope on two GPUs.

Set SCOPE=full (or another supported scope) before the command to measure it.
These measurements use two RTX GPUs and must not be presented as four-H100 timings.
Old 2026-10-27 JSON reports remain historical and used the older 2.6B schedule.

All day estimates assume two GPUs running a balanced queue continuously, no
retries, and approximately stable throughput. The additional 20% planning
allowance is not a confidence interval. Short startup measurements do not test
sustained thermal behavior or long-job storage variation. Neither proposed
scenario changes the current formal research protocol.
