"""Check the release snapshot against the workspace's authoritative document."""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
source=a.source or root.parent/'nanogpt_experiment_design.md'
if source.read_text(encoding='utf8').strip()!=(root/'docs/nanogpt_experiment_design.md').read_text(encoding='utf8').strip():
    raise SystemExit('Design snapshot mismatch: update source first, then synchronize snapshot')
print('Design snapshot matches authoritative source (line endings normalized)')
