"""Read-only acceptance checks before a GPU launch."""
import argparse
from dataclasses import asdict
import json
import torch
from .data import verify
from .model import Shape
from .protocol import Budget


def check_hardware(micro, expected_gpus=1, training_world=1):
    """Validate visible hardware separately from the training world size."""
    if expected_gpus < 1 or training_world < 1:
        raise ValueError('expected_gpus and training_world must be positive')
    if training_world > expected_gpus:
        raise ValueError('training_world cannot exceed expected_gpus')
    accumulation = Budget().accumulations(training_world, micro)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: activate an environment with CUDA PyTorch')
    visible = torch.cuda.device_count()
    if visible != expected_gpus:
        raise RuntimeError(f'Expected {expected_gpus} visible GPU(s), found {visible}; '
                           'check the allocation and CUDA_VISIBLE_DEVICES')
    if training_world > 1 and not torch.distributed.is_nccl_available():
        raise RuntimeError('NCCL unavailable in this PyTorch build')
    names = []
    for index in range(visible):
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f'GPU {index} does not support BF16')
            names.append(torch.cuda.get_device_name(index))
    return dict(gpus=names, visible_gpus=visible, training_world=training_world,
                micro=micro, accumulation=accumulation,
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
    parser.add_argument('--expected-gpus', type=int, default=1)
    parser.add_argument('--training-world', type=int, default=1)
    args = parser.parse_args()
    report = check_hardware(args.micro_batch, args.expected_gpus, args.training_world)
    print('Checking frozen data and artifact hashes...', flush=True)
    report['artifact_id'] = check_inputs(args.data, args.artifacts)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
