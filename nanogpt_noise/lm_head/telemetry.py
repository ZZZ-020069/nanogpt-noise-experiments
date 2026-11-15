"""Head-only cloud observation; the frozen last-k uploader stays unchanged."""
import importlib
import json
import os

from ..telemetry import Telemetry as BaseTelemetry


def check_environment():
    enabled=os.environ.get('NANOGPT_SWANLAB','0')
    if enabled not in ('0','1'):
        raise ValueError('NANOGPT_SWANLAB must be 0 or 1')
    if enabled=='0':
        return dict(enabled=False)
    if not os.environ.get('SWANLAB_API_KEY','').strip():
        raise RuntimeError('NANOGPT_SWANLAB=1 requires SWANLAB_API_KEY in the task environment')
    try:
        sdk=importlib.import_module('swanlab')
    except Exception:
        raise RuntimeError('Install a working SwanLab SDK in the selected Python environment') from None
    return dict(enabled=True, sdk_version=str(getattr(sdk,'__version__','unknown')),
                project=os.environ.get('SWANLAB_PROJ_NAME','nanogpt-noise'),
                cloud_authentication='not_checked')


class Telemetry(BaseTelemetry):
    def log(self,step,tokens,metrics,evaluations,state):
        if self.process is None:
            return
        values=dict(metrics)
        values.update(running=int(state=='running'),
                      infrastructure_failure=int(state=='infrastructure_failure'))
        if evaluations:
            latest=evaluations[-1]
            values['objective_eval_step']=latest['step']
            for key in ('train_objective','val_objective'):
                if key in latest:
                    values[key]=latest[key]
        super().log(step,tokens,values,evaluations,state)


if __name__=='__main__':
    print(json.dumps(check_environment(),indent=2))
