"""Optional cloud observation in an isolated process; local files remain authoritative."""
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import uuid


def payload(step,tokens,metrics,evaluations,state):
    values={'progress/tokens':tokens,'progress/step':step,
            'status/completed':int(state=='completed'),
            'status/numerical_failure':int(state=='numerical_failure')}
    for k,v in metrics.items():
        if isinstance(v,(int,float)) and math.isfinite(v): values['diagnostic/'+k]=v
    if evaluations:
        e=evaluations[-1]
        for k in ('step','tokens','train','val'):
            v=e.get(k)
            if isinstance(v,(int,float)) and math.isfinite(v): values['evaluation/'+k]=v
    return values


def worker(messages,config,name,directory):
    parent=mp.parent_process()
    if parent is not None:
        def watch_parent():
            while parent.is_alive():
                threading.Event().wait(2)
            os._exit(0)
        threading.Thread(target=watch_parent,daemon=True).start()
    try:
        import swanlab
        run=swanlab.init(project=os.environ.get('SWANLAB_PROJ_NAME','nanogpt-noise'),
                         workspace=os.environ.get('SWANLAB_WORKSPACE') or None,
                         experiment_name=name,config=config,mode='online',
                         logdir=directory)
        print('[swanlab] initialized '+name,flush=True)
        sequence=0
        while True:
            item=messages.get()
            if item is None: break
            run.log(item,step=sequence)
            sequence+=1
        run.finish()
        print('[swanlab] upload session finished '+name,flush=True)
    except Exception:
        # SDK exceptions may contain authentication material: never echo them.
        print('[swanlab] upload failed; local training records retained',flush=True)


class Telemetry:
    def __init__(self,output,config,identity,mode,start_step):
        self.process=None;self.messages=None;self.warned=False
        if mode=='test' or os.environ.get('NANOGPT_SWANLAB')!='1': return
        if not os.environ.get('SWANLAB_API_KEY'):
            raise RuntimeError('NANOGPT_SWANLAB=1 requires SWANLAB_API_KEY')
        import importlib.util
        if importlib.util.find_spec('swanlab') is None:
            raise RuntimeError('Install SwanLab before enabling telemetry')
        attempt=uuid.uuid4().hex[:12]
        output=Path(output)
        base=Path(os.environ.get('RUN_BASE',str(output.parent)))
        directory=base/'swanlab'/mode/attempt
        directory.mkdir(parents=True,exist_ok=False)
        name=f'{mode}-{config["scope"]}-{config["optimizer"]}-n{config["node"]}-{output.name}-{attempt}'
        context=mp.get_context('spawn');self.messages=context.Queue(maxsize=64)
        self.process=context.Process(target=worker,args=(self.messages,
            dict(**config,training_run_id=identity,attempt=attempt,resumed_from_step=start_step,mode=mode),
            name,str(directory)),daemon=False)
        self.process.start()

    def log(self,step,tokens,metrics,evaluations,state):
        if self.process is None: return
        try:
            if not self.process.is_alive(): raise queue.Full
            self.messages.put_nowait(payload(step,tokens,metrics,evaluations,state))
        except queue.Full:
            if not self.warned:
                print('[swanlab] uploader unavailable or behind; cloud points may be missing; local records retained',flush=True)
                self.warned=True

    def close(self):
        if self.process is None: return
        try: self.messages.put_nowait(None)
        except queue.Full: pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=2)
            print('[swanlab] upload flush timed out; local records retained',flush=True)
        self.messages.cancel_join_thread();self.messages.close()
