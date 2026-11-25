from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass
class Cohort:
    ids: np.ndarray
    motor: np.ndarray
    concepts: np.ndarray
    concept_names: np.ndarray
    labels: np.ndarray | None
    genes: np.ndarray | None = None
    gene_rows: np.ndarray | None = None
    gene_names: np.ndarray | None = None


def strings(value, name):
    if value.ndim != 1 or value.dtype.kind not in 'US':
        raise ValueError(f'{name} must be a one-dimensional string array')
    value = value.astype(str)
    if len(set(value.tolist())) != len(value) or np.any(value == ''):
        raise ValueError(f'{name} must contain unique nonempty values')
    return value


def numeric(value, name, binary=False):
    if value.dtype.kind not in 'bifu' or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite numeric values')
    if binary and not np.isin(value, [0, 1]).all():
        raise ValueError(f'{name} must contain only 0 and 1')
    return value


@contextmanager
def open_arrays(path):
    path = Path(path)
    if path.is_dir():
        class Arrays(dict):
            @property
            def files(self):
                return list(self)
        names = ['ids', 'motor', 'concepts', 'concept_names', 'labels', 'genes', 'gene_ids', 'gene_names']
        yield Arrays({name: np.load(path / (name + '.npy'), allow_pickle=False)
                      for name in names if (path / (name + '.npy')).is_file()})
    else:
        with np.load(path, allow_pickle=False) as arrays:
            yield arrays


def load_cohort(path, require_labels=True, require_genes=False):
    with open_arrays(path) as f:
        required = {'ids', 'motor', 'concepts', 'concept_names'}
        if require_labels:
            required.add('labels')
        if require_genes:
            required.update(['genes', 'gene_ids', 'gene_names'])
        if not required.issubset(f.files):
            raise ValueError('Missing required arrays: ' + ', '.join(sorted(required - set(f.files))))
        ids = strings(f['ids'], 'ids')
        motor = numeric(f['motor'], 'motor').astype(np.float32)
        concepts = numeric(f['concepts'], 'concepts', binary=True).astype(np.float32)
        names = strings(f['concept_names'], 'concept_names')
        n = len(ids)
        if n == 0 or motor.ndim != 2 or concepts.ndim != 2 or motor.shape[0] != n or concepts.shape != (n, len(names)) or not motor.shape[1] or not len(names):
            raise ValueError('Invalid patient, MOTOR or concept dimensions')
        labels = None
        if 'labels' in f:
            labels = numeric(f['labels'], 'labels', binary=True).astype(np.int64)
            if labels.shape != (n,):
                raise ValueError('labels must match ids row order')
        cohort = Cohort(ids, motor, concepts, names, labels)
        if 'genes' in f or require_genes:
            if not {'genes', 'gene_ids', 'gene_names'}.issubset(f.files):
                raise ValueError('genes, gene_ids and gene_names must be supplied together')
            gene_ids = strings(f['gene_ids'], 'gene_ids')
            gene_names = strings(f['gene_names'], 'gene_names')
            gene_values = numeric(f['genes'], 'genes')
            if (gene_values < 0).any():
                raise ValueError('genes must contain nonnegative counts or binary values')
            genes = (gene_values > 0).astype(np.float32)
            if genes.shape != (len(gene_ids), len(gene_names)) or not len(gene_names):
                raise ValueError('Invalid gene dimensions')
            lookup = {pid: i for i, pid in enumerate(gene_ids)}
            if not set(lookup).issubset(set(ids.tolist())):
                raise ValueError('gene_ids must be a subset of ids')
            rows = np.array([lookup.get(pid, -1) for pid in ids], dtype=np.int64)
            cohort.genes, cohort.gene_rows, cohort.gene_names = genes, rows, gene_names
        return cohort


def graph_edges(concepts, device):
    patients, concepts_idx = np.nonzero(concepts)
    targets = concepts_idx + len(concepts)
    pairs = np.stack([patients, targets], axis=1)
    edges = np.stack([pairs, pairs[:, ::-1]], axis=1).reshape(-1, 2).T
    return torch.as_tensor(edges, dtype=torch.long, device=device)


def source_split(labels, seed):
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 5:
        raise ValueError('Source splitting needs at least five examples per class')
    development, test = next(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))
    train, validation = train_test_split(development, test_size=0.25, stratify=labels[development], random_state=seed)
    return {'train': train, 'validation': validation, 'test': test}


def target_split(labels, budget, seed):
    if budget < 10 or budget >= len(labels):
        raise ValueError('Target labeled budget must be at least ten and smaller than the cohort')
    development, test = train_test_split(np.arange(len(labels)), train_size=budget, stratify=labels, random_state=seed)
    train, validation = train_test_split(development, test_size=0.2, stratify=labels[development], random_state=seed)
    return {'train': train, 'validation': validation, 'test': test}


def cohort_fingerprint(ids):
    return hashlib.sha256(json.dumps(ids.tolist(), ensure_ascii=True).encode()).hexdigest()
