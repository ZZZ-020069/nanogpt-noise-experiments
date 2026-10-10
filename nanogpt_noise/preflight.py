"""Read-only acceptance checks before a single-node four-GPU launch."""
import argparse
from dataclasses import asdict
import json
import torch
from .data import verify
from .model import Shape
from .protocol import Budget


def check_hardware(micro):
    accumulation = Budget().accumulations(4, micro)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: activate an environment with CUDA PyTorch')
    if torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly four visible GPUs required; allocate four GPUs or set CUDA_VISIBLE_DEVICES')
    if not torch.distributed.is_nccl_available():
        raise RuntimeError('NCCL unavailable in this PyTorch build')
    names = []
    for index in range(4):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f'GPU {index} does not support BF16')
            names.append(torch.cuda.get_device_name(index))
    return dict(gpus=names, world=4, micro=micro, accumulation=accumulation,
                global_batch=512, tokens_per_step=524288,
                torch=str(torch.__version__), cuda=torch.version.cuda)


def check_inputs(data, artifacts):
    identity = verify(data, artifacts)
    if identity['tiny']:
        raise ValueError('Refusing tiny test artifacts for a real-model launch')
    if identity['budget'] != asdict(Budget()) or identity['shape'] != asdict(Shape()):
        raise ValueError('Frozen artifacts do not match the current protocol')
    return identity['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--micro-batch', type=int, default=4)
    args = parser.parse_args()
    report = check_hardware(args.micro_batch)
    print('Checking frozen data and artifact hashes...', flush=True)
    report['artifact_id'] = check_inputs(args.data, args.artifacts)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
