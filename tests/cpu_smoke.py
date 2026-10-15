"""Real tiny-model CPU smoke, never a formal run or GPU acceptance marker."""
import argparse
from pathlib import Path
import platform
import sys
import torch
from nanogpt_noise.artifacts import write_json
from nanogpt_noise.protocol import configuration, grid
from nanogpt_noise.train import Runtime, run
from nanogpt_noise.node import run_node
from .test_integration import fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Refuse overwrite: these synthetic inputs must never replace real data.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, artifacts = fixture(args.output)
    runtime = Runtime(cpu=True)
    trajectories = []
    for scope in ('full', 'last1'):
        for optimizer in ('adamw', 'adam', 'sgd', 'muon'):
            config = configuration(scope, optimizer, 10, 0)
            output = args.output / f'{scope}-{optimizer}'
            result = run(config, data, artifacts, output, micro=2,
                         runtime=runtime, test=True)
            assert result['status'] == 'completed', result
            assert result['mode'] == 'test' and result['step'] == 4
            trajectories.append(dict(scope=scope, optimizer=optimizer,
                                     loss=result['loss'], steps=result['step']))
    node = run_node(grid()[0], data, artifacts, args.output / 'eight-candidate-node',
                    micro=2, runtime=runtime, test=True)
    assert node['complete'] and node['valid_count'] == 8, node
    report = dict(kind='tiny_cpu_smoke_not_h100_acceptance',
                  python=sys.version, torch=str(torch.__version__),
                  platform=platform.platform(), trajectories=trajectories,
                  node_candidates=len(node['candidates']),
                  valid_candidates=node['valid_count'])
    write_json(args.output / 'cpu_smoke_summary.json', report)
    print('CPU SMOKE PASSED (synthetic tiny model, not H100 acceptance)')
    print(args.output / 'cpu_smoke_summary.json')


if __name__ == '__main__':
    main()
