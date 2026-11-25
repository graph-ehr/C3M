"""Independent synthetic arrays; never reads user data or fitted statistics."""
from pathlib import Path
import numpy as np


def generate(output, seed=7, source_n=240, target_n=180, format="npz"):
    if format not in ("npz", "separate"):
        raise ValueError("format must be npz or separate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for tag, n in [('source', source_n), ('target', target_n)]:
        rng = np.random.default_rng(seed + int(tag == 'target'))
        motor = rng.normal(size=(n, 12)).astype(np.float32)
        concepts = rng.binomial(1, 0.3, (n, 8)).astype(np.float32)
        genes = rng.binomial(1, 0.3, (n, 6)).astype(np.float32)
        risk = motor[:, 0] + 0.4 * concepts[:, 0] + 0.5 * genes[:, 0] + rng.normal(0, 0.4, n)
        labels = (risk > np.quantile(risk, 0.6)).astype(np.int64)
        ids = np.array([f'toy_{tag}_{i:04d}' for i in range(n)])
        arrays = dict(ids=ids, motor=motor, concepts=concepts, labels=labels,
                      concept_names=np.array([f'concept_{i:02d}' for i in range(8)]))
        if tag == 'source':
            rows = rng.permutation(n)[:int(n * 0.8)]
            arrays.update(genes=genes[rows], gene_ids=ids[rows],
                          gene_names=np.array([f'gene_{i:02d}' for i in range(6)]))
        if format == 'npz':
            np.savez_compressed(output / f'{tag}.npz', **arrays)
        else:
            directory = output / tag
            directory.mkdir()
            for name, array in arrays.items():
                np.save(directory / (name + '.npy'), array, allow_pickle=False)
