"""Search-level recovery with real tiny models; no GPU or production inputs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from nanogpt_noise.artifacts import read_json, file_hash, write_json
from nanogpt_noise.node import run_node
from nanogpt_noise.protocol import grid, configuration
from nanogpt_noise.train import run, Runtime
from tests.test_integration import fixture


class SearchResumeTests(unittest.TestCase):
    def test_pause_request_saves_step_and_resumes_with_same_optimizer_state(self):
        from nanogpt_noise import train
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            data,artifacts=fixture(root)
            rt=Runtime(cpu=True)
            cfg=configuration('last1','muon',10,0.)
            pause=root/'pause.request'
            original=train.inject
            def request(*args,**kwargs):
                pause.touch()
                return original(*args,**kwargs)
            with patch.object(train,'inject',side_effect=request):
                result=run(cfg,data,artifacts,root/'paused',micro=2,runtime=rt,test=True,pause_file=pause)
            self.assertEqual(result['step'],1)
            self.assertEqual(result['status'],'infrastructure_failure')
            held=run(cfg,data,artifacts,root/'paused',micro=2,runtime=rt,test=True,pause_file=pause)
            self.assertEqual(held['step'],1)
            pause.unlink()
            actual=run(cfg,data,artifacts,root/'paused',micro=2,runtime=rt,test=True,pause_file=pause)
            expected=run(cfg,data,artifacts,root/'whole',micro=2,runtime=rt,test=True)
            self.assertEqual(actual['loss'],expected['loss'])
            a=torch.load(root/'paused/final.pt',weights_only=True)
            b=torch.load(root/'whole/final.pt',weights_only=True)
            def equal(x,y):
                if isinstance(x,torch.Tensor): self.assertTrue(torch.equal(x,y))
                elif isinstance(x,dict):
                    self.assertEqual(x.keys(),y.keys())
                    for key in x: equal(x[key],y[key])
                elif isinstance(x,(list,tuple)):
                    self.assertEqual(len(x),len(y))
                    for left,right in zip(x,y): equal(left,right)
                else: self.assertEqual(x,y)
            equal(a['optimizer'],b['optimizer'])
            equal(a['model'],b['model'])

    def test_rollback_discards_uncommitted_diagnostics_and_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            data,artifacts=fixture(root)
            rt=Runtime(cpu=True)
            cfg=configuration('last1','adamw',10,0.)
            output=root/'run'
            run(cfg,data,artifacts,output,micro=2,runtime=rt,test=True,pause_after=1)
            for step in (2,3,4):
                write_json(output/'steps'/f'{step:06d}.json',dict(step=step,stale=True))
            write_json(output/'evaluations.json',[dict(step=999)])
            pause=root/'pause.request';pause.touch()
            held=run(cfg,data,artifacts,output,micro=2,runtime=rt,test=True,pause_file=pause)
            self.assertEqual(held['step'],1)
            self.assertEqual([p.name for p in (output/'steps').glob('*.json')],['000001.json'])
            self.assertEqual(read_json(output/'evaluations.json')[0]['step'],0)
            (output/'latest.pt').unlink()
            with self.assertRaisesRegex(RuntimeError,'checkpoint missing'):
                run(cfg,data,artifacts,output,micro=2,runtime=rt,test=True)
            (output/'status.json').unlink()  # hard kill may never publish status
            with self.assertRaisesRegex(RuntimeError,'checkpoint missing'):
                run(cfg,data,artifacts,output,micro=2,runtime=rt,test=True)

    def test_search_pause_resume_preserves_pending_and_final_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, artifacts = fixture(root)
            rt = Runtime(cpu=True)
            for job in (grid()[140+i*35+10] for i in range(4)):
                with self.subTest(optimizer=job['optimizer']):
                    whole = root/(job['optimizer']+'-whole')
                    resumed = root/(job['optimizer']+'-resumed')
                    expected = run_node(job,data,artifacts,whole,micro=2,runtime=rt,test=True)
                    def interrupt(config, data, artifacts, output, **kwargs):
                        if output.name == 'candidate_01':
                            kwargs['pause_after'] = 1
                        return run(config,data,artifacts,output,**kwargs)
                    with patch('nanogpt_noise.node.run',side_effect=interrupt):
                        paused = run_node(job,data,artifacts,resumed,micro=2,runtime=rt,test=True)
                    self.assertEqual(paused['status'],'infrastructure_failure')
                    pending = resumed/'candidate_01/latest.pt'
                    self.assertEqual(torch.load(pending,weights_only=True)['step'],1)
                    def checked(config, data, artifacts, output, **kwargs):
                        if output.name == 'candidate_01':
                            self.assertTrue(pending.exists(),'replay deleted the pending checkpoint')
                            self.assertEqual(torch.load(pending,weights_only=True)['step'],1)
                        return run(config,data,artifacts,output,**kwargs)
                    with patch('nanogpt_noise.node.run',side_effect=checked):
                        actual = run_node(job,data,artifacts,resumed,micro=2,runtime=rt,test=True)
                    self.assertEqual(actual,expected)
                    winner=actual['winner']['candidate']
                    path=resumed/f'candidate_{winner:02d}/final.pt'
                    saved=torch.load(path,weights_only=True)
                    original=torch.load(whole/f'candidate_{winner:02d}/final.pt',weights_only=True)
                    for name,value in saved['model'].items():
                        self.assertTrue(torch.equal(value,original['model'][name]))
                    checksum=file_hash(path)
                    self.assertEqual(run_node(job,data,artifacts,resumed,micro=2,runtime=rt,test=True),actual)
                    self.assertEqual(file_hash(path),checksum)

    def test_completed_search_replay_preserves_later_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            data,artifacts=fixture(root)
            rt=Runtime(cpu=True)
            job=grid()[140]
            def decreasing(config,data,artifacts,output,**kwargs):
                result=run(config,data,artifacts,output,**kwargs)
                # Force a later winner independently of incidental tiny-model loss order.
                from nanogpt_noise.artifacts import write_json
                result['loss']=10-int(output.name[-2:])
                write_json(output/'status.json',result)
                return result
            output=root/'search'
            with patch('nanogpt_noise.node.run',side_effect=decreasing):
                result=run_node(job,data,artifacts,output,micro=2,runtime=rt,test=True)
                self.assertEqual(result['winner']['candidate'],7)
                path=output/'candidate_07/final.pt'
                before=file_hash(path)
                run_node(job,data,artifacts,output,micro=2,runtime=rt,test=True)
                self.assertTrue(path.exists(),'replay deleted the final winner')
                self.assertEqual(file_hash(path),before)
