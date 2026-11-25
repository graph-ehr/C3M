# C3M

This repository provides the implementation of C3M from the paper **Democratizing Clinical Risk Prediction with Cross-Cohort Cross-Modal Knowledge Transfer**, published at NeurIPS 2025.
C3M learns from EHR and genetic data in a source cohort and supports prediction and
adaptation in target cohorts with EHR data alone. The model combines graph-guided EHR
representation adaptation, a gene Transformer with learnable aggregation tokens, and
teacher–student knowledge distillation.

This repository provides the main method and synthetic example. The synthetic example uses independently
generated random data. Clinical data are not distributed.

## Environment requirements

The implementation was tested with:

- Python 3.10.16
- NumPy 1.26.4
- scikit-learn 1.1.1
- PyTorch 2.5.1+cu121
- PyTorch Geometric 2.6.1

Run examples from the repository directory. CPU execution is supported; GPU execution
requires a compatible GPU driver and CUDA-enabled PyTorch build.

## Toy example

If `toy-data/` is included in your download, use it directly. Otherwise generate it once:

```bash
python -m c3m toy --output toy-data
```

The source and target examples use the same input interface as real data. Each variable
is stored in a separate `.npy` file:

```text
toy-data/
  source/
    ids.npy
    motor.npy
    concepts.npy
    concept_names.npy
    labels.npy
    genes.npy
    gene_ids.npy
    gene_names.npy
  target/
    ids.npy
    motor.npy
    concepts.npy
    concept_names.npy
    labels.npy
```

Train on the synthetic source cohort. This command automatically trains the model,
selects the best checkpoint using validation AUROC, and evaluates it on the test set:

```bash
python -m c3m train \
  --data toy-data/source \
  --output runs/source \
  --device cpu \
  --epochs 2 \
  --batch-size 32
```

Fine-tune the source checkpoint on the synthetic target cohort using EHR alone.
This command also performs validation and final test evaluation automatically:

```bash
python -m c3m finetune \
  --data toy-data/target \
  --checkpoint runs/source/best.pt \
  --output runs/target \
  --labeled-budget 60 \
  --device cpu \
  --epochs 2 \
  --batch-size 32
```

Use a new output location for each experiment. Toy results demonstrate the workflow and
have no clinical interpretation.

## Preparing real data

Provide the same files as the toy example. 

| Array | Shape | Description |
| --- | --- | --- |
| `ids` | `[N]` | Unique, nonempty patient identifiers |
| `motor` | `[N, D]` | Pretrained EHR representations |
| `concepts` | `[N, C]` | Binary diagnosis/medication concept indicators |
| `concept_names` | `[C]` | Unique concept names in column order |
| `labels` | `[N]` | Binary outcomes for training and evaluation |
| `genes` | `[G, K]` | Observed binary gene indicators or nonnegative counts for source training |
| `gene_ids` | `[G]` | Unique subset of `ids`, aligned to gene rows |
| `gene_names` | `[K]` | Unique gene names in column order |

Source training requires EHR, labels and observed genetics. Target fine-tuning requires
EHR and labels.
