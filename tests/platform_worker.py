"""CPU-only subprocess fixture for platform termination tests."""
import argparse
from pathlib import Path
import time
from unittest.mock import patch
import torch
from nanogpt_noise import train
from nanogpt_noise.artifacts import lock
from nanogpt_noise.protocol import configuration
from tests.test_integration import fixture


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--reference',action='store_true')
    args=parser.parse_args()
    rt=train.Runtime(cpu=True)
    root=args.root
    data,artifacts=root/'data',root/'artifacts'
    if not data.exists():
        fixture(root)
    original=train.inject
    def slow_step(*a,**kw):
        result=original(*a,**kw)
        (root/'active').write_text(str(a[3]))
        time.sleep(.6)
        return result
    with lock(root/'scope/launcher'),lock(root/'scope/job_0140'):
        with patch.object(train,'inject',side_effect=original if args.reference else slow_step):
            result=train.run(configuration('last1','muon',10,0.),data,artifacts,
                             root/('reference' if args.reference else 'scope/job_0140/candidate_00'),
                             micro=2,runtime=rt,test=True)
    raise SystemExit(75 if result['status']=='infrastructure_failure' else 0)


if __name__=='__main__': main()
