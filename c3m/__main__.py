import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .data import load_cohort, cohort_fingerprint
from .engine import Config, train, predict
from .metrics import classification
from .toy import generate


def training_arguments(parser):
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    for name, default in [('epochs', 300), ('patience', 30), ('batch-size', 128),
                          ('seed', 42), ('split-seed', 42), ('tokens', 2), ('threads', 4)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('lr', .001), ('student-lr', .0001), ('dropout', .1),
                          ('gene-dropout', .1), ('beta', .1), ('kd-weight', .01), ('temperature', 1.5)]:
        parser.add_argument('--' + name, type=float, default=default)


def main():
    parser = argparse.ArgumentParser(prog='c3m', description='C3M training and EHR-only deployment')
    sub = parser.add_subparsers(dest='command', required=True)
    toy = sub.add_parser('toy', help='Generate independent synthetic source/target arrays')
    toy.add_argument('--output', required=True)
    toy.add_argument('--seed', type=int, default=7)
    toy.add_argument('--format', choices=['separate', 'npz'], default='separate')
    source = sub.add_parser('train', help='Train source teacher/student with observed genetics')
    training_arguments(source)
    target = sub.add_parser('finetune', help='Adapt encoder/student to labeled target development patients')
    training_arguments(target)
    target.add_argument('--checkpoint', required=True)
    target.add_argument('--labeled-budget', type=int, default=500)
    for command in ['predict', 'evaluate']:
        p = sub.add_parser(command, help='EHR-only inference; no teacher or genetics required')
        p.add_argument('--data', required=True)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--output', required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--seed', type=int, default=42)
        if command == 'evaluate':
            p.add_argument('--splits', help='Saved splits.npz; omit to evaluate the whole supplied cohort')
            p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    args = parser.parse_args()
    if args.command == 'toy':
        generate(args.output, args.seed, format=args.format)
        print('Synthetic source and target files generated.')
    elif args.command in ['train', 'finetune']:
        cfg = Config(**{key: getattr(args, key) for key in Config.__dataclass_fields__})
        data = load_cohort(args.data, require_genes=args.command == 'train')
        train(data, args.output, cfg, args.device,
              source_checkpoint=getattr(args, 'checkpoint', None), labeled_budget=getattr(args, 'labeled_budget', 500))
    else:
        output = Path(args.output)
        if output.exists():
            raise ValueError('Output already exists')
        data = load_cohort(args.data, require_labels=args.command == 'evaluate')
        scores = predict(data, args.checkpoint, args.device, args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'predict':
            with output.open('x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'probability'])
                writer.writerows(zip(data.ids.tolist(), scores.tolist()))
        else:
            rows = np.arange(len(data.ids))
            if args.splits:
                with np.load(args.splits, allow_pickle=False) as splits:
                    if 'cohort_fingerprint' not in splits or str(splits['cohort_fingerprint'].item()) != cohort_fingerprint(data.ids):
                        raise ValueError('Saved splits belong to a different cohort or patient order')
                    rows = splits[args.split]
                if rows.ndim != 1 or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows) or (rows < 0).any() or (rows >= len(data.ids)).any():
                    raise ValueError('Invalid saved split indices')
            y, p = data.labels[rows], scores[rows]
            p = np.clip(p, 1e-7, 1 - 1e-7)
            w = np.where(y == 1, 10., 1.)
            ce = float(np.sum(w * (-y * np.log(p) - (1-y) * np.log1p(-p))) / w.sum())
            result = classification(y, p, (p > 0.5).astype(int), ce)
            output.write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(result))


if __name__ == '__main__':
    main()
