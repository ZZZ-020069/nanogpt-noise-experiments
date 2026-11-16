"""Offline curvature workflow. Fit and held-out checks are separate commands."""
from dataclasses import asdict
from pathlib import Path
import math
import os
import re
import shlex
import sys
import tempfile
import time
import numpy as np
import torch
from ..artifacts import (read_json, write_json, save_checkpoint, file_hash,
                         tensor_hash, digest, lock)
from ..data import verify, batch
from ..input_bundle import verify_bundle
from ..model import Shape
from .model import HeadGPT
from .protocol import VERSION, Budget, code_identity, require_current_input, DEFAULT_MICRO, require_micro
from .curvature import (probabilities, gradient, trace, hessian_block,
                        statistics_block, sample_blocks, block_metrics,
                        inclusion_moments)
from .train import Runtime
from .diagnostics import compatible, singular_values, taylor_chunk, fixed_gradient_probe


def load_statistics(path, identity, shape, allow_test=False):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if (artifact.get('schema') != 'lm-head-K-v1'
            or artifact.get('design') != VERSION
            or artifact.get('input_id') != identity['id']
            or artifact.get('initialization_hash') != identity['initialization_tensor_hash']
            or artifact.get('shape') != asdict(shape)
            or artifact.get('code') != code_identity()['sha256']
            or artifact.get('passed') is not True
            or artifact.get('test') != allow_test):
        raise ValueError('K statistics must pass fit and held-out checks for this input/code/mode')
    stats = artifact['statistics']
    if stats['B'].shape != (shape.n_embd, shape.n_embd) or stats['pbar'].shape != (shape.vocab_size,):
        raise ValueError('statistics shape mismatch')
    if not all(torch.isfinite(stats[k]).all() for k in ('B', 'pbar', 'alpha')):
        raise ValueError('nonfinite statistics')
    return stats


def probe_windows(length, sequence, count, tiny=False):
    """Nested prefixes; reserve the complete fit pool before drawing check."""
    reserve = max(count, 16 if tiny else 128)
    used, pools = [], {}
    for name, seed in [('fit', 2026111401), ('check', 2026111402)]:
        gen = torch.Generator().manual_seed(seed)
        positions = []
        for _ in range(reserve*10000):
            at = int(torch.randint(length-sequence, (), generator=gen))
            if all(abs(at-other) > sequence for other in used):
                used.append(at)
                positions.append(at)
                if len(positions) == reserve:
                    break
        if len(positions) != reserve:
            raise ValueError('corpus too small for disjoint diagnostic window pools')
        pools[name] = positions[:count]
    return pools


def _statistics(b, pbar, t, n):
    b, pbar, t = b/n, pbar/n, t/n
    denominator = (1-pbar.square().sum())*b.trace()
    if denominator <= 0:
        raise ValueError('degenerate curvature')
    return dict(B=b.cpu(), pbar=pbar.cpu(), alpha=(t/denominator).cpu(), trace=t.cpu())


def measure(input_root, output, mode='fit', *, tokens=32768, count=None,
            fit_dir=None, checkpoint=None, chunk_tokens=64, micro=DEFAULT_MICRO, test=False):
    require_micro(micro,test)
    started = time.monotonic()
    count = (2 if checkpoint else 5) if count is None else count
    root, output = Path(input_root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError('diagnostic output and inputs must be separate trees')
    data, artifacts = root/'openwebtext_subset', root/'frozen-artifacts'
    ident = verify(data, artifacts)
    if ident['tiny'] != test:
        raise ValueError('tiny fixtures and formal diagnostic modes cannot mix')
    if not test:
        verify_bundle(root)
        require_current_input(ident)
        if tokens not in (32768, 65536, 131072) or count not in ((2,) if checkpoint else (5, 10)):
            raise ValueError('formal probes use 32768/65536/131072 tokens and 5/10 fit blocks')
    if mode not in ('fit', 'check'):
        raise ValueError('mode must be fit or check')
    if mode == 'check' and (not fit_dir or count != (2 if checkpoint else 5)):
        raise ValueError('held-out check requires a frozen fit and the prescribed block count')
    budget, shape = Budget(**ident['budget']), Shape(**ident['shape'])
    budget.accumulations(1,micro)
    if tokens % budget.sequence or tokens <= 0 or chunk_tokens < 1 or micro < 1:
        raise ValueError('probe tokens must be a positive multiple of sequence length')
    rt = Runtime(cpu=test)
    if rt.world != 1:
        raise ValueError('offline measurement is single-device')
    model = HeadGPT(shape)
    model.load_state_dict(torch.load(artifacts/'initial.pt', weights_only=True, map_location='cpu'))
    if tensor_hash(model.named_parameters()) != ident['initialization_tensor_hash']:
        raise ValueError('initialization hash mismatch')
    model.activate_head()
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    frozen_hash = tensor_hash(frozen)
    source_hash = None
    if checkpoint:
        source_hash = file_hash(checkpoint)
        state = torch.load(checkpoint, weights_only=True, map_location='cpu')
        if state.get('input_id') != ident['id'] or state.get('backbone_hash') != frozen_hash:
            raise ValueError('checkpoint is not a Head-only checkpoint for these inputs')
        with torch.no_grad():
            model.lm_head.weight.copy_(state['head'])
    model.to(rt.device).eval()
    positions = probe_windows(ident['lengths']['train'], budget.sequence,
                              tokens//budget.sequence, test)[mode]
    chosen = [b for b in sample_blocks(shape.vocab_size, shape.n_embd, tiny=test, count=max(5,count))
              if b['pool'] == mode and b['index'] < count]
    execution = dict(design=VERSION, input_id=ident['id'], code=code_identity()['sha256'],
                     source_checkpoint_sha256=source_hash, mode=mode, tokens=tokens,
                     blocks=chosen, windows=positions, test=test, chunk_tokens=chunk_tokens,
                     micro=micro, torch=str(torch.__version__), cuda=torch.version.cuda,
                     fit_sha256=file_hash(Path(fit_dir)/'fit.pt') if mode=='check' else None)
    run_id = digest(execution)
    parent = None
    if mode == 'check':
        parent = torch.load(Path(fit_dir)/'fit.pt',weights_only=True,map_location='cpu')
        if (parent['input_id'] != ident['id'] or parent['code'] != code_identity()['sha256']
                or parent['source_checkpoint_sha256'] != source_hash
                or parent['tokens'] != tokens or (not parent['fit_passed'] and source_hash is None)
                or parent['test'] != test):
            raise ValueError('held-out check requires a compatible passing frozen fit')
    with lock(output):
        if (output/'identity.json').exists():
            if read_json(output/'identity.json')['id'] != run_id:
                raise ValueError('diagnostic output is bound to different inputs/settings')
            if (output/'report.json').exists():
                return read_json(output/'report.json')
        elif any(p.name not in ('.lock', '.lease') for p in output.iterdir()):
            raise ValueError('refusing a nonempty unregistered diagnostic directory')
        write_json(output/'identity.json', dict(id=run_id, **execution))
        array = np.memmap(data/'train.bin', dtype='<u2', mode='r')
        hs, ys = [], []
        try:
            with torch.no_grad():
                for begin in range(0, len(positions), micro):
                    x, y = batch(array, positions[begin:begin+micro], budget.sequence, rt.device)
                    with rt.autocast():
                        h = model.features(x)
                    hs.append(h.float().flatten(0, 1).cpu())
                    ys.append(y.flatten().cpu())
        finally:
            array._mmap.close()
        h, y = torch.cat(hs), torch.cat(ys)
        save_checkpoint(output/'features.pt', dict(h=h, y=y, windows=positions, input_id=ident['id']))
        w = model.lm_head.weight.detach().double()
        blocks = [torch.zeros(len(b['rows'])*len(b['cols']), len(b['rows'])*len(b['cols']),
                              dtype=torch.float64, device=rt.device) for b in chosen]
        bsum = torch.zeros(shape.n_embd, shape.n_embd, dtype=torch.float64, device=rt.device)
        psum = torch.zeros(shape.vocab_size, dtype=torch.float64, device=rt.device)
        gsum = torch.zeros_like(w)
        tsum = torch.zeros((), dtype=torch.float64, device=rt.device)
        tgsum = torch.zeros_like(w)
        losses = dict(real=0.,initial=0.,taylor=0.)
        w0 = model.reference_head.double()
        prefix_size = 8192 if not test else max(budget.sequence, tokens//4)
        prefix = None
        done = 0
        while done < tokens:
            end = min(tokens, done+chunk_tokens)
            if done < prefix_size < end:
                end = prefix_size
            x = h[done:end].to(rt.device, dtype=torch.float64)
            target = y[done:end].to(rt.device)
            p = probabilities(x, w)
            n = len(x)
            bsum += x.T @ x
            psum += p.sum(0)
            tsum += trace(x, p)*n
            gsum += gradient(x, p, target)*n
            diagnostics = taylor_chunk(x,target,w,w0)
            tgsum += diagnostics['taylor_gradient']*n
            for key in losses:
                losses[key] += float(diagnostics[key])*n/tokens
            for definition, value in zip(chosen, blocks):
                value += hessian_block(x, p, definition['rows'], definition['cols'])*n
            done = end
            if done == prefix_size:
                prefix = dict(g=(gsum/done).cpu(), blocks=[
                    block_metrics(value/done)[0] for value in blocks])
            print(f'curvature {mode}: {done}/{tokens} tokens', flush=True)
        stats = _statistics(bsum, psum, tsum, tokens)
        g = (gsum/tokens).cpu()
        if mode == 'check':
            comparison = parent['statistics']
        else:
            comparison = stats
        reports, layout_errors = [], {0: [], 1: []}
        stable = {0: [], 1: []}
        for index, (definition, total) in enumerate(zip(chosen, blocks)):
            block = (total/tokens).cpu()
            gs = g[definition['rows']][:, definition['cols']]
            metrics, eigen = block_metrics(block, gs)
            r, c = len(definition['rows']),len(definition['cols'])
            row_ids = torch.arange(r).repeat_interleave(c)
            cross = row_ids[:,None] != row_ids[None,:]
            energy = float(block.square().sum())
            metrics['cross_row_energy_fraction'] = float(block[cross].square().sum())/energy if energy else None
            metrics['diagonal_quantiles'] = torch.quantile(block.diag(),torch.tensor(
                [0.,.05,.5,.95,1.],dtype=torch.float64)).tolist()
            estimate = statistics_block(comparison, definition['rows'], definition['cols'])
            scale = float(torch.linalg.vector_norm(block))
            discrepancy = float(torch.linalg.vector_norm(block-estimate))
            error = discrepancy/scale if scale else (0. if discrepancy==0 else float('inf'))
            layout_errors[definition['layout']].append(error)
            changes = {}
            if prefix:
                for key in ('trace', 'maximum', 'second_moment'):
                    large, small = metrics[key], prefix['blocks'][index][key]
                    changes[key] = abs(large-small)/abs(large) if large else (0. if small == 0 else float('inf'))
                if definition['index'] < 5:
                    stable[definition['layout']].append(changes['trace'] <= .1
                        and changes['maximum'] <= .1 and changes['second_moment'] <= .2)
            item = dict(**definition, metrics=metrics, relative_error=error, changes=changes,
                        full_parameter_moments=inclusion_moments(block, definition['rows'],
                            definition['cols'], shape.vocab_size, shape.n_embd))
            reports.append(item)
            save_checkpoint(output/f'block-{index:02d}.pt',
                            dict(definition=definition, hessian=block, gradient=gs, eigenvalues=eigen))
        norm = float(torch.linalg.vector_norm(g))
        pg = prefix['g'] if prefix else g
        pn = float(torch.linalg.vector_norm(pg))
        cosine = float((g*pg).sum())/(norm*pn) if norm*pn else (1. if not norm and not pn else 0.)
        norm_change = abs(norm-pn)/norm if norm else (0. if not pn else float('inf'))
        stability = all(sum(v) >= min(4, len(v)) for v in stable.values()) and norm_change <= .1 and cosine >= .99
        geometry = all(float(torch.quantile(torch.tensor(v,dtype=torch.float64),.5)) <= .2
                       and max(v) <= .5 for v in layout_errors.values())
        compatibility = compatible(comparison, g)
        numeric = all(r['metrics']['psd'] and r['metrics']['symmetry_error'] <= 1e-10 for r in reports)
        trace_error = abs(float(comparison['trace']-stats['trace']))/float(stats['trace'])
        indices = torch.load(artifacts/'windows.pt',weights_only=True,map_location='cpu')
        fixed = fixed_gradient_probe(model,data,indices,budget,rt,micro,comparison,mode,test,
                                     chunk_tokens=chunk_tokens)
        write_json(output/'fixed-gradients.json',fixed)
        passed = (numeric and geometry and compatibility['passed'] and fixed['passed']
                  and (stability if mode=='fit' else trace_error <= .1))
        if source_hash is not None:
            passed = numeric  # Later points measure drift, not eligibility of the initial proxy.
        error_g = float(torch.linalg.vector_norm(g-tgsum.cpu()/tokens))
        error_l = abs(losses['real']-losses['taylor'])
        displacement = float(torch.linalg.vector_norm(w-w0))
        moment_summary = {}
        for layout in (0,1):
            selected = [b['full_parameter_moments'] for b in reports if b['layout']==layout]
            moment_summary[layout] = {}
            for key in ('trace','second_moment'):
                values = torch.tensor([b[key] for b in selected],dtype=torch.float64)
                moment_summary[layout][key] = dict(mean=float(values.mean()),
                    std=float(values.std()) if len(values)>1 else None,
                    minimum=float(values.min()),maximum=float(values.max()))
            moment = moment_summary[layout]['second_moment']['mean']
            moment_summary[layout]['effective_rank_plugin'] = float(stats['trace'])**2/moment if moment else None
        report = dict(id=run_id, mode=mode, input_id=ident['id'], tokens=tokens, test=test,
                      trace=float(stats['trace']), trace_relative_error=trace_error,
                      gradient_norm=norm, gradient_cosine_prefix=cosine,
                      gradient_norm_change=norm_change, stability_passed=stability,
                      geometry_passed=geometry, compatibility=compatibility, numeric_passed=numeric,
                      fixed_gradient_checks_passed=fixed['passed'],
                      gradient_singular_values=singular_values(g), moments_by_layout=moment_summary,
                      taylor=dict(**losses,displacement=displacement,absolute_loss_error=error_l,
                          relative_loss_change_error=error_l/abs(losses['real']-losses['initial'])
                            if abs(losses['real']-losses['initial'])>1e-12 else None,
                          absolute_gradient_error=error_g,
                          relative_gradient_error=error_g/norm if norm>1e-12 else None),
                      elapsed_seconds=time.monotonic()-started,
                      purpose='trajectory_drift' if source_hash else 'initial_proxy_acceptance',
                      passed=passed, blocks=reports,
                      inference='Block spectra are local; weighted moments target this fixed empirical Hessian.')
        record = dict(design=VERSION, code=code_identity()['sha256'], input_id=ident['id'],
                      initialization_hash=ident['initialization_tensor_hash'], shape=asdict(shape),
                      source_checkpoint_sha256=source_hash, tokens=tokens, test=test,
                      statistics=stats, fit_passed=passed)
        if mode == 'fit':
            save_checkpoint(output/'fit.pt', record)
        elif passed and source_hash is None:
            save_checkpoint(output/'K-approved.pt', dict(
                schema='lm-head-K-v1', **{k: parent[k] for k in ('design','code','input_id',
                'initialization_hash','shape','test','statistics')}, passed=True,
                fit_sha256=file_hash(Path(fit_dir)/'fit.pt'), check_id=run_id))
        write_json(output/'report.json', report)
        return report


_ACTIVE_K_RE = re.compile(r'^\s*(?:export\s+)?K_STATISTICS=(.*)$')
_COMMENT_K_RE = re.compile(r'^\s*#\s*(?:export\s+)?K_STATISTICS=.*$')


def _parse_env_value(raw):
    raw = raw.strip()
    if not raw or raw.startswith('#'):
        return ''
    try:
        tokens = shlex.split(raw, comments=True, posix=True)
        return tokens[0] if tokens else ''
    except ValueError:
        return raw.strip('"\'')


def sync_k_statistics_env(env_file, statistics_path, *, checkpoint=None, report=None, force=False):
    """Persist approved initial K-approved.pt into a sourced .env file without touching drift checks."""
    if not env_file or checkpoint is not None:
        return 'skipped'
    if report is not None and (not report.get('passed') or report.get('purpose') != 'initial_proxy_acceptance'):
        return 'skipped'
    env_path, stats_path = Path(env_file), Path(statistics_path)
    if not env_path.is_file() or not stats_path.is_file():
        return 'skipped'
    resolved = str(stats_path.resolve())
    export_line = f'export K_STATISTICS={shlex.quote(resolved)}'
    text = env_path.read_text(encoding='utf8')
    lines = text.splitlines(keepends=True)

    active_idx = None
    existing_val = ''
    comment_idx = None
    for idx, line in enumerate(lines):
        m = _ACTIVE_K_RE.match(line.rstrip('\r\n'))
        if m:
            active_idx = idx
            existing_val = _parse_env_value(m.group(1))
            break
        if comment_idx is None and _COMMENT_K_RE.match(line.rstrip('\r\n')):
            comment_idx = idx

    if active_idx is not None and existing_val:
        if os.path.normcase(str(Path(existing_val).resolve())) == os.path.normcase(resolved):
            return 'unchanged'
        if not force:
            print(f'[env-sync] K_STATISTICS already active in {env_path}; keeping {existing_val} (set LH_FORCE_K_ENV=1 to overwrite).',
                  file=sys.stderr)
            return 'preserved'

    target_idx = active_idx if active_idx is not None else comment_idx
    if target_idx is not None:
        eol = '\r\n' if lines[target_idx].endswith('\r\n') else ('\n' if lines[target_idx].endswith('\n') else '')
        lines[target_idx] = export_line + (eol or '\n')
    else:
        if lines and not lines[-1].endswith(('\n', '\r')):
            lines[-1] = lines[-1] + '\n'
        lines.append(export_line + '\n')

    mode = env_path.stat().st_mode
    fd, tmp_name = tempfile.mkstemp(prefix='.env-sync-', dir=str(env_path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf8', newline='') as handle:
            handle.write(''.join(lines))
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, env_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    print(f'[env-sync] Updated K_STATISTICS={resolved} in {env_path}', file=sys.stderr)
    return 'updated'
