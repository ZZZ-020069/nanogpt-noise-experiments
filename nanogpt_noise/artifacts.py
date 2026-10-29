"""Atomic outputs, provenance and cooperative single-writer locks."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import uuid


def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def tensor_hash(named):
    h = hashlib.sha256()
    for n,p in named:
        h.update(n.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def safe_json(value):
    import math
    if isinstance(value,float) and not math.isfinite(value):
        return str(value)
    if isinstance(value,dict):
        return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [safe_json(v) for v in value]
    return value


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path,value):
    atomic(path,lambda f:f.write(json.dumps(safe_json(value),indent=2,ensure_ascii=False,allow_nan=False).encode('utf8')))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save_checkpoint(path,value):
    import torch
    atomic(path,lambda f:torch.save(value,f))


def cleanup_atomic_temps(directory):
    """Only call under the enclosing writer lock; match our mkstemp names."""
    names={'.lock','latest.pt','final.pt','failure.pt','initial.pt','windows.pt',
           'metadata.json','status.json','evaluations.json','identity.json','meta.json',
           'search.json','steps.jsonl.gz','steps.compaction.json','preparation.json',
           'registration.json','SHA256SUMS','scope_run.json','scope_manifest.json'}
    for path in Path(directory).glob('*.tmp'):
        match=re.fullmatch(r'(.+)\.[a-z0-9_]{8}\.tmp',path.name)
        if (match and (match[1] in names or re.fullmatch(r'\d{6}\.json',match[1]))
                and path.is_file()):
            path.unlink()


@contextlib.contextmanager
def lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/'.lock'
    if os.name == 'posix':
        import fcntl
        # Never unlink this inode: replacement would split writers across locks.
        with (directory/'.lease').open('a+b') as lease:
            try:
                fcntl.flock(lease.fileno(),fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'{path}: active or stale lock; another writer owns the kernel lease') from exc
            except OSError as exc:
                raise RuntimeError(f'{directory}: filesystem must support reliable flock; refusing unsafe lock fallback') from exc
            try:
                if path.exists():
                    try:
                        owner=read_json(path)
                    except (ValueError,OSError) as exc:
                        raise RuntimeError(f'{path}: legacy or unknown lock; verify no worker remains before manual recovery') from exc
                    if not isinstance(owner,dict) or owner.get('format') != 'nanogpt-flock-v1':
                        raise RuntimeError(f'{path}: legacy or unknown lock; manual recovery required')
                    print(f'Recovering abandoned lock marker: {path}',flush=True)
                cleanup_atomic_temps(directory)
                write_json(path,dict(format='nanogpt-flock-v1',host=platform.node(),
                                     pid=os.getpid(),token=uuid.uuid4().hex))
                try:
                    yield
                finally:
                    path.unlink(missing_ok=True)
            finally:
                fcntl.flock(lease.fileno(),fcntl.LOCK_UN)
        return
    # Windows test fallback: no PID/hostname guessing or automatic stale cleanup.
    try:
        fd = os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'{path}: active or stale lock; verify no worker remains before removing it')
    try:
        with os.fdopen(fd,'w') as f:
            f.write(f'{platform.node()} pid={os.getpid()}\n')
        yield
    finally:
        path.unlink(missing_ok=True)


def code_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'nanogpt_noise').glob('*.py'))
    files += sorted((root/'docs').glob('nanogpt_experiment_design.md'))
    hashes = {p.relative_to(root).as_posix():file_hash(p) for p in files}
    return dict(sha256=digest(hashes),files=hashes)


def environment(device):
    import torch
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        commit = None
    driver=None
    if device.type=='cuda':
        try:
            driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                            '--format=csv,noheader'],text=True,timeout=5).strip()
        except (OSError,subprocess.SubprocessError):
            pass
    return dict(python=platform.python_version(),platform=platform.platform(),
                torch=str(torch.__version__),cuda=torch.version.cuda,commit=commit,
                device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                driver=driver,gpu_memory=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else None,
                cudnn=torch.backends.cudnn.version(),code=code_identity())
