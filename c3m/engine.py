from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import graph_edges, source_split, target_split, cohort_fingerprint
from .metrics import classification
from .models import EHRGNN_motor, TeacherTransformerModel, StudentModel


@dataclass
class Config:
    seed: int = 42
    split_seed: int = 42
    epochs: int = 300
    patience: int = 30
    batch_size: int = 128
    lr: float = 0.001
    student_lr: float = 0.0001
    dropout: float = 0.1
    gene_dropout: float = 0.1
    tokens: int = 2
    beta: float = 0.1
    kd_weight: float = 0.01
    temperature: float = 1.5
    threads: int = 4

    def validate(self):
        if min(self.epochs, self.patience, self.batch_size, self.threads) < 1:
            raise ValueError('epochs, patience, batch size and threads must be positive')
        if not all(np.isfinite(x) and x > 0 for x in [self.lr, self.student_lr, self.temperature]):
            raise ValueError('Learning rates and temperature must be finite and positive')
        if not 0 <= self.dropout < 1 or not 0 <= self.gene_dropout < 1:
            raise ValueError('Dropout must be in [0, 1)')
        if not np.isfinite(self.beta) or self.beta < 0 or not 0 <= self.kd_weight <= 1 or not 1 <= self.tokens <= 4:
            raise ValueError('Invalid reconstruction weight, KD weight or token count')


def setup(config, device):
    config.validate()
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device(device)


def make_models(data, config, device):
    motor = torch.as_tensor(data.motor, device=device)
    encoder = EHRGNN_motor(motor, len(data.ids) + len(data.concept_names), 16, 16, 16, config.dropout).to(device)
    student = StudentModel(ehr_dim=16).to(device)
    return encoder, student, graph_edges(data.concepts, device)


def batch_rows(rows, size, shuffle=False):
    # Training uses the standard PyTorch sampler/global RNG, as in the development loader.
    # Monitoring has its own generator so extra metric passes do not change training order.
    generator = None if shuffle else torch.Generator().manual_seed(0)
    loader = DataLoader(np.asarray(rows, dtype=np.int64), batch_size=size,
                        shuffle=shuffle, num_workers=0, generator=generator)
    for batch in loader:
        yield batch.numpy()


@torch.no_grad()
def evaluate(data, encoder, student, edges, rows, size, device, teacher=None):
    encoder.eval()
    student.eval()
    if teacher is not None:
        teacher.eval()
        rows = rows[data.gene_rows[rows] >= 0]
    if not len(rows):
        raise ValueError('No eligible evaluation patients')
    embeddings = encoder(edges)
    scores, predictions, targets = [], [], []
    numerator = denominator = 0.0
    weights = torch.tensor([1., 10.], device=device)
    for r in batch_rows(rows, size):
        ehr = embeddings[r]
        if teacher is None:
            logits, _ = student(ehr)
        else:
            genes = torch.as_tensor(data.genes[data.gene_rows[r]], device=device)
            logits, _, _, _ = teacher(ehr, genes)
        y = torch.as_tensor(data.labels[r], device=device)
        numerator += F.cross_entropy(logits, y, weight=weights, reduction='sum').item()
        denominator += weights[y].sum().item()
        scores.extend(logits.softmax(1)[:, 1].cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    result = classification(np.asarray(targets), np.asarray(scores), np.asarray(predictions), numerator / denominator)
    result['n'] = len(rows)
    return result


def save_history(output, history):
    (output / 'history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
    try:
        from .plotting import plot_history
        plot_history(history, output)
    except ImportError:
        pass


def student_objective(logits, labels, paired_positions, teacher_logits, weights, config):
    ce = F.cross_entropy(logits, labels, weight=weights)
    kd = logits.new_zeros(())
    if len(paired_positions):
        kd = F.kl_div(F.log_softmax(logits[paired_positions] / config.temperature, dim=1),
                      F.softmax(teacher_logits / config.temperature, dim=1).clamp_min(1e-8),
                      reduction='batchmean') * config.temperature**2
    return (1 - config.kd_weight) * ce + config.kd_weight * kd, ce, kd


def checkpoint(config, data, encoder, student, teacher, epoch):
    return {'format_version': 1, 'config': asdict(config), 'motor_dim': data.motor.shape[1],
            'concept_names': data.concept_names.tolist(),
            'gene_names': data.gene_names.tolist() if data.gene_names is not None else [],
            'encoder': encoder.state_dict(), 'student': student.state_dict(),
            'teacher': teacher.state_dict() if teacher is not None else None, 'epoch': epoch}


def restore(path, data, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    if payload['motor_dim'] != data.motor.shape[1]:
        raise ValueError('MOTOR representation width differs from checkpoint')
    if payload['concept_names'] != data.concept_names.tolist():
        raise ValueError('Concept vocabulary/order differs from checkpoint')
    config = Config(**payload['config'])
    encoder, student, edges = make_models(data, config, device)
    encoder.load_state_dict(payload['encoder'], strict=True)
    student.load_state_dict(payload['student'], strict=True)
    return payload, encoder, student, edges


def train(data, output, config, device='cpu', source_checkpoint=None, labeled_budget=500):
    device = setup(config, device)
    output = Path(output)
    if data.labels is None:
        raise ValueError('Training requires labels')
    source = source_checkpoint is None
    if source and (data.genes is None or data.gene_rows is None):
        raise ValueError('Source training requires observed gene data')
    splits = source_split(data.labels, config.split_seed) if source else target_split(data.labels, labeled_budget, config.split_seed)
    for rows in splits.values():
        if len(np.unique(data.labels[rows])) != 2:
            raise ValueError('Each split must contain both classes')
    if source:
        paired_train = splits['train'][data.gene_rows[splits['train']] >= 0]
        if len(np.unique(data.labels[paired_train])) != 2:
            raise ValueError('Paired training subset must contain both classes')
        encoder, student, edges = make_models(data, config, device)
        teacher = TeacherTransformerModel(gene_dim=len(data.gene_names), num_masks=config.tokens, gene_dropout=config.gene_dropout).to(device)
        teacher_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(teacher.parameters()), lr=config.lr, eps=1e-5)
        optimizer = torch.optim.Adam(student.parameters(), lr=config.student_lr, eps=1e-5)
    else:
        _, encoder, student, edges = restore(source_checkpoint, data, device)
        encoder.dropout = config.dropout
        teacher = None
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(student.parameters()), lr=config.lr, eps=1e-5)
    # Refuse to overwrite existing experiments, including failed or incomplete outputs.
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'splits.npz', **splits, cohort_fingerprint=np.array(cohort_fingerprint(data.ids)))
    (output / 'config.json').write_text(json.dumps(asdict(config), indent=2))
    history, best, stale = [], -float('inf'), 0
    best_epoch = -1
    weights = torch.tensor([1., 10.], device=device)
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch}/{config.epochs}...", flush=True)
        objective, steps = 0.0, 0
        train_total = train_main = train_gene = 0.0
        counts = dict(student_ce_n=0, gene_free_ce_n=0, kd_n=0, teacher_n=0)
        for step, r in enumerate(batch_rows(splits['train'], config.batch_size, True)):
            encoder.train()
            student.train()
            embeddings = encoder(edges)[r]
            labels = torch.as_tensor(data.labels[r], device=device)
            if source:
                paired_positions = np.flatnonzero(data.gene_rows[r] >= 0)
                teacher_loss = classification_loss = reconstruction = embeddings.new_zeros(())
                teacher_logits = None
                if len(paired_positions):
                    teacher.train()
                    paired_rows = r[paired_positions]
                    genes = torch.as_tensor(data.genes[data.gene_rows[paired_rows]], device=device)
                    paired_embeddings = embeddings[paired_positions]
                    logits, _, rec_logits, mask = teacher(paired_embeddings, genes)
                    rec = F.cross_entropy(rec_logits.flatten(0, 1), genes.flatten().long(), reduction='none')
                    reconstruction = (rec * mask.reshape(-1)).sum() / mask.sum()
                    classification_loss = F.cross_entropy(logits, labels[paired_positions], weight=weights)
                    teacher_loss = classification_loss + config.beta * reconstruction
                    if not torch.isfinite(teacher_loss):
                        raise ValueError('Nonfinite teacher objective')
                    teacher_optimizer.zero_grad()
                    teacher_loss.backward()
                    teacher_optimizer.step()
                    teacher.eval()
                    with torch.no_grad():
                        teacher_logits = teacher(paired_embeddings.detach(), genes)[0]
                student_logits = student(embeddings.detach())[0]
                loss, student_ce, kd = student_objective(student_logits, labels, paired_positions,
                                                        teacher_logits, weights, config)
                counts['student_ce_n'] += len(r)
                counts['gene_free_ce_n'] += len(r) - len(paired_positions)
                counts['kd_n'] += len(paired_positions)
                counts['teacher_n'] += len(paired_positions)
            else:
                loss = F.cross_entropy(student(embeddings)[0], labels, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            objective += loss.item()
            train_total += teacher_loss.item() if source else loss.item()
            train_main += classification_loss.item() if source else loss.item()
            train_gene += reconstruction.item() if source else 0.0
            steps += 1
        if not steps:
            raise ValueError('No eligible training batches')
        if epoch % 2 == 0:
            message = f"Epoch {epoch} | Train loss: {train_total / steps:.4f} | Train main loss: {train_main / steps:.4f}"
            if source:
                message += f" | Train gene loss: {train_gene / steps:.4f}"
            print(message, flush=True)
        tr = evaluate(data, encoder, student, edges, splits['train'], config.batch_size, device)
        val = evaluate(data, encoder, student, edges, splits['validation'], config.batch_size, device)
        item = {'epoch': epoch, 'supervision_counts': counts, 'objective': objective / steps, 'student_train': tr, 'student_validation': val}
        if teacher is not None:
            # Teacher curves require both classes among observed-genetics patients.
            for key in ['train', 'validation']:
                paired = splits[key][data.gene_rows[splits[key]] >= 0]
                if len(np.unique(data.labels[paired])) == 2:
                    item[f'teacher_{key}'] = evaluate(data, encoder, student, edges, paired, config.batch_size, device, teacher)
        history.append(item)
        save_history(output, history)
        labels = [('Validation Loss', 'ce'), ('AUROC', 'auroc'), ('auprc', 'auprc'),
                  ('ACC', 'acc'), ('F1binary', 'f1'), ('sensi90', 'sensitivity_at_90'),
                  ('sensi95', 'sensitivity_at_95'), ('ppv90', 'ppv_at_90'), ('ppv95', 'ppv_at_95')]
        for label, key in labels:
            print(f"\tEpoch {epoch} | Step {step} | {label}: {val[key]:.4f}", flush=True)
        if val['auroc'] > best:
            best, stale = val['auroc'], 0
            best_epoch = epoch
            torch.save(checkpoint(config, data, encoder, student, teacher, epoch), output / 'best.pt')
            print(f"New best model at epoch {epoch}, AUROC: {best:.4f}", flush=True)
        else:
            stale += 1
            print(f"No improvement. Early stop counter: {stale}/{config.patience}", flush=True)
            print(f"-- best model at epoch {best_epoch}, AUROC: {best:.4f}", flush=True)
        if stale >= config.patience:
            print("Early stopping triggered.", flush=True)
            break
    payload, encoder, student, edges = restore(output / 'best.pt', data, device)
    metrics = {'best_epoch': payload['epoch'], 'best_validation': history[payload['epoch']]['student_validation'],
               'test': evaluate(data, encoder, student, edges, splits['test'], config.batch_size, device)}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print('Test metrics:', json.dumps(metrics['test']), flush=True)
    return metrics


@torch.no_grad()
def predict(data, checkpoint_path, device='cpu', seed=42):
    setup(Config(seed=seed), device)
    device = torch.device(device)
    _, encoder, student, edges = restore(checkpoint_path, data, device)
    encoder.eval()
    student.eval()
    # No teacher or gene arrays are accessed here.
    return student(encoder(edges)[:len(data.ids)])[0].softmax(1)[:, 1].cpu().numpy()
