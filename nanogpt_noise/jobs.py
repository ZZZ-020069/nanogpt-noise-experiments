"""Export the complete job manifest without launching compute."""
import argparse
from .protocol import grid,Budget,VERSION
from .artifacts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    write_json(a.output,dict(design=VERSION,jobs=[dict(job_index=i,**j) for i,j in enumerate(grid())],
                           candidates=6720,tokens_per_candidate=Budget().tokens,
                           maximum_base_tokens=6720*Budget().tokens))
    print('840 configurations; 6720 candidate trajectories; no jobs launched')


if __name__=='__main__': main()
