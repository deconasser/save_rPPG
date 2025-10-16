#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eda.py — EDA HR for MMPD (per record & per subject), mirroring model.py test logic
----------------------------------------------------------------------------------
- Concatenate chunks per (subject, record) using chunk_interval (same as model.py)
- Compute GT HR using your `metrics.calculate_hr_and_hrv` (defaults only; fs=30, bpmmin=40, bpmmax=180)
- Aggregate per-subject stats
- Default split: test
- Accepts --split_idx and writes it into CSVs (useful when your CSV marks fold)

Outputs:
- <output_dir>/gt_hr_per_record_<split>_fold<split_idx>.csv
- <output_dir>/gt_hr_per_subject_<split>_fold<split_idx>.csv
"""

import os
import re
import csv
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from datasets import get_dataset_cls, collate_fn
from metrics import calculate_hr_and_hrv  # uses defaults: fs=30, bpmmin=40, bpmmax=180


# ------------------------ Helpers ------------------------

def _is_diff_wave(config: Dict[str, Any]) -> bool:
    try:
        wt = config['data'].get('wave_type', [])
        if isinstance(wt, (list, tuple)) and len(wt) > 0:
            return 'diff' in str(wt[0]).lower()
        return 'diff' in str(wt).lower()
    except Exception:
        return False


def _concat_record_chunks(idx_to_wave: Dict[int, np.ndarray], chunk_interval: int) -> np.ndarray:
    if not idx_to_wave:
        return np.asarray([], dtype=np.float32)
    parts = []
    for i, idx in enumerate(sorted(idx_to_wave.keys())):
        w = idx_to_wave[idx]
        if i > 0 and chunk_interval is not None and chunk_interval > 0:
            w = w[-chunk_interval:]
        parts.append(w)
    cat = np.concatenate(parts, axis=0)
    cat = np.asarray(cat).squeeze()
    if cat.ndim == 2 and cat.shape[1] > 1:
        cat = cat[:, 0]
    return cat.astype(np.float32, copy=False)


def _build_loaders(config: Dict[str, Any], split: str) -> List[DataLoader]:
    assert split in {'train','val','test'}
    ds_cfg = config['data']
    batch_size = ds_cfg['batch_size']
    num_workers = ds_cfg['num_workers']

    if split == 'train':
        train_sets = []
        for args in ds_cfg['train_sets']:
            ds_cls = get_dataset_cls(args['name'])
            train_sets.append(
                ds_cls(**ds_cfg['datasets'][args['name']],
                       split=args['split'],
                       split_idx=config['split_idx'],
                       training=True)
            )
        concat = ConcatDataset(train_sets)
        return [DataLoader(concat, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=False,
                           drop_last=False, persistent_workers=False,
                           collate_fn=collate_fn)]
    key = 'val_sets' if split=='val' else 'test_sets'
    loaders = []
    for args in ds_cfg.get(key, []):
        ds_cls = get_dataset_cls(args['name'])
        ds = ds_cls(**ds_cfg['datasets'][args['name']],
                    split=args['split'],
                    split_idx=config['split_idx'],
                    training=False)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=False,
                                  persistent_workers=False, collate_fn=collate_fn))
    return loaders


_fname_pat = re.compile(r'(?P<subject>\d+)[-_](?P<record>\d+)[-_](?P<idx>\d+)', re.IGNORECASE)

def _infer_sri_from_filename(meta: Dict[str, Any]) -> Tuple[str, str, int]:
    """
    Fallback: infer (subject, record, idx) from filename like '4-0-12.pt'
    """
    for key in ['filename','file','path','name']:
        val = meta.get(key)
        if isinstance(val, str):
            base = os.path.basename(val)
            m = _fname_pat.search(base)
            if m:
                return m.group('subject'), m.group('record'), int(m.group('idx'))
    # if not found, create generic ones
    return meta.get('subject',''), meta.get('record',''), int(meta.get('idx', 0))


# ------------------------ Core EDA ------------------------

def run_hr_eda(config: Dict[str, Any], split: str='test', split_idx: int=0, output_dir: str='analysis') -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    is_diff = _is_diff_wave(config)
    chunk_interval = int(config['data']['chunk_interval'])
    loaders = _build_loaders(config, split)

    # Collect GT per chunk like in model.py:test_step
    gt_map: Dict[int, Dict[str, Dict[str, Dict[int, np.ndarray]]]] = {}
    for dl_id, loader in enumerate(loaders):
        for batch in loader:
            frames, waves, meta = batch
            B = frames.shape[0] if hasattr(frames, 'shape') else len(meta)
            gt_map.setdefault(dl_id, {})
            for i in range(B):
                m = meta[i]
                subject = m.get('subject', None)
                record = m.get('record', None)
                idx = m.get('idx', None)
                if subject is None or record is None or idx is None:
                    subject, record, idx = _infer_sri_from_filename(m)

                gt_map[dl_id].setdefault(subject, {}).setdefault(record, {})[idx] = waves[i].detach().cpu().numpy()

    # Per-record HR using your function (defaults only)
    per_record_rows = []
    subj_to_bpms: Dict[str, List[float]] = {}
    for dl_id, by_subj in gt_map.items():
        for subject, by_rec in by_subj.items():
            for record, by_idx in by_rec.items():
                gt = _concat_record_chunks(by_idx, chunk_interval)
                if gt.size == 0:
                    continue
                measures = calculate_hr_and_hrv(gt, diff=is_diff)  # defaults: fs=30, bpmmin=40, bpmmax=180
                bpm = float(measures[0]) if (measures is not None and len(measures)>0 and np.isfinite(measures[0])) else float('nan')
                dur_sec = float(len(gt)/30.0)  # duration assuming fs=30 to match defaults
                per_record_rows.append((split, split_idx, dl_id, subject, record, len(by_idx), dur_sec, 30.0, bpm))
                if np.isfinite(bpm):
                    subj_to_bpms.setdefault(subject, []).append(bpm)

    # Write per-record CSV
    per_rec_csv = os.path.join(output_dir, f'gt_hr_per_record_{split}_fold{split_idx}.csv')
    with open(per_rec_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split','fold','dataloader_id','subject','record','n_chunks','duration_sec','fs_hz','bpm_estimate'])
        for row in per_record_rows:
            sp, fold, dl, subj, rec, n_chunks, dur, fs_hz, bpm = row
            w.writerow([sp, fold, dl, subj, rec, n_chunks, f'{dur:.3f}', f'{fs_hz:.3f}', f'{bpm:.3f}' if np.isfinite(bpm) else 'nan'])

    # Aggregate per-subject
    per_subj_csv = os.path.join(output_dir, f'gt_hr_per_subject_{split}_fold{split_idx}.csv')
    with open(per_subj_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split','fold','subject','n_records','mean_bpm','std_bpm','min_bpm','max_bpm','median_bpm','p10_bpm','p90_bpm'])
        for subject, bpms in subj_to_bpms.items():
            arr = np.asarray(bpms, dtype=float)
            if arr.size == 0:
                continue
            w.writerow([
                split, split_idx, subject, arr.size,
                f'{np.mean(arr):.3f}', f'{np.std(arr):.3f}', f'{np.min(arr):.3f}', f'{np.max(arr):.3f}',
                f'{np.median(arr):.3f}', f'{np.percentile(arr,10):.3f}', f'{np.percentile(arr,90):.3f}'
            ])
    return per_rec_csv, per_subj_csv


# ------------------------ CLI ------------------------

def _load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def _load_config_from_module(module_path: str) -> Dict[str, Any]:
    mod = __import__(module_path, fromlist=['*'])
    if not hasattr(mod, 'config'):
        raise AttributeError(f"Module '{module_path}' must expose a top-level 'config' dict.")
    return getattr(mod, 'config')


def main():
    p = argparse.ArgumentParser(description='EDA HR (per record & subject) — mirrors model.py test logic')
    p.add_argument('--config', type=str, default=None, help='Path to YAML config')
    p.add_argument('--config-module', type=str, default=None, help='Python module exposing dict `config`')
    p.add_argument('--split', type=str, default='test', choices=['train','val','test'])
    p.add_argument('--split_idx', type=int, default=0, help='Fold index for logging/filenames (no impact on data selection here)')
    p.add_argument('--output-dir', type=str, default='analysis')
    args = p.parse_args()

    if not args.config and not args.config_module:
        raise SystemExit('Provide --config <yaml> or --config-module <module.path>')

    config = _load_config_from_yaml(args.config) if args.config else _load_config_from_module(args.config_module)

    # ensure split_idx in config so dataset classes that rely on it still work
    config['split_idx'] = args.split_idx

    # basic validations
    for k in ['data','split_idx']:
        if k not in config:
            raise KeyError(f"Missing config key: {k}")
    if 'chunk_interval' not in config['data']:
        raise KeyError("Missing config['data']['chunk_interval']")

    per_rec_csv, per_subj_csv = run_hr_eda(config, split=args.split, split_idx=args.split_idx, output_dir=args.output_dir)
    print('Wrote:')
    print(' -', per_rec_csv)
    print(' -', per_subj_csv)


if __name__ == '__main__':
    main()
