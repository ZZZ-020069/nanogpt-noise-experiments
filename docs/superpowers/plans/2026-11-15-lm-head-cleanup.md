# LM Head fixed-layout cleanup

Goal: retain only the formal micro16 workflow in active code and documentation.
Execution: inline, as requested. Authority: main design section 14.22.

- Remove the selector, legacy benchmark workers/wrapper, empirical K module and dedicated tests. Remove benchmark branches from train.py.
- Add one require_micro validator to protocol.py and call it before formal CLI, trainer, root binding, worker-root validation and curvature work. Formal micro must equal 16; explicit CPU fixtures retain valid small layouts.
- Update shell routing and tests to reject other micro values. Keep full 200-sequence weighted evaluation and gradient/Hessian tests, using 8 as the comparison chunk where needed.
- Archive prior selection evidence and snapshots by exact file rename into history; preserve immutable raw JSON and all remote data. Rewrite active README, operational guide and implementation summaries around micro16 only.
- Run all retained LM Head math, recovery, micro and shell/H100 tests; parse documented Bash. Check baseline source and archived raw result hashes.
- Publish fresh immutable tag lm-head-h100-m16-20261115-v2, with new code/output paths. Do not move the old release tag or touch deployed jobs.

Validator contract:

~~~python
def require_micro(micro, test=False):
    bound_micro(dict(micro=micro))
    if not test and micro != DEFAULT_MICRO:
        raise ValueError('LM Head execution requires micro=16')
~~~

Validation: all 33 retained LM Head tests passed in 61.435 seconds; the final default-layout test rerun passed. Bash syntax and actual H100 plan routing checked. Historical files match the prior commit byte for byte; the original last-k design prefix is unchanged. No GPU task was started.
