"""Validate all four optimizer smoke outputs before a matching scope launch."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from .artifacts import code_identity, digest, read_json
from .protocol import Budget, CENTERS, SCOPES, configuration, require_current_input


def validate(runs, scope, input_id, micro):
    for optimizer in CENTERS:
        path = Path(runs)/'smoke'/'1gpu'/f'{scope}-{optimizer}-node10'
        status, metadata = read_json(path/'status.json'), read_json(path/'metadata.json')
        execution = metadata['execution']
        expected = dict(config=configuration(scope, optimizer, 10, 0.0), mode='smoke',
                        artifact_id=input_id, code=code_identity()['sha256'], world=1,
                        micro=micro, device_type='cuda', torch=str(torch.__version__),
                        cuda=torch.version.cuda, target_steps=8)
        if execution != expected or metadata['run_id'] != digest(expected):
            raise ValueError(f'{optimizer}: smoke config/code/input/runtime/layout mismatch; rerun smoke')
        require_current_input(metadata['artifact_identity'])
        if metadata['artifact_identity']['id'] != input_id:
            raise ValueError(f'{optimizer}: smoke input mismatch')
        expected_d = 124373760 if scope == 'full' else 7079424*int(scope[4:])
        if metadata['D'] != expected_d:
            raise ValueError(f'{optimizer}: smoke trainable dimension mismatch')
        if any(status.get(key) != value for key, value in
               dict(status='completed', mode='smoke', step=8, target_steps=8,
                    tokens=8*Budget().step_tokens, run_id=metadata['run_id']).items()):
            raise ValueError(f'{optimizer}: smoke did not complete eight steps')
        if not (path/'final.pt').is_file():
            raise ValueError(f'{optimizer}: smoke final checkpoint missing')
        evaluations = read_json(path/'evaluations.json')
        if [row['step'] for row in evaluations] != [0, 8]:
            raise ValueError(f'{optimizer}: smoke evaluation endpoints missing')
    return dict(scope=scope, optimizers=list(CENTERS), steps=8, input_id=input_id,
                budget=asdict(Budget()), micro=micro)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--scope', choices=SCOPES, required=True)
    parser.add_argument('--input-id', required=True)
    parser.add_argument('--micro-batch', type=int, default=4)
    args = parser.parse_args()
    print(validate(args.runs, args.scope, args.input_id, args.micro_batch))


if __name__ == '__main__':
    main()
