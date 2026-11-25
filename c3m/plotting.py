import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history, output):
    for name in ['student', 'teacher']:
        available = [row for row in history if f'{name}_train' in row and f'{name}_validation' in row]
        if not available:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for ax, metric in zip(axes, ['ce', 'auroc', 'auprc']):
            for split in ['train', 'validation']:
                ax.plot([r['epoch'] for r in available], [r[f'{name}_{split}'][metric] for r in available], label=split)
            ax.set(xlabel='Epoch (zero-based)', ylabel=metric.upper())
            ax.grid(alpha=0.2)
            ax.legend()
        scope = 'observed genetics only' if name == 'teacher' else 'all patients'
        fig.suptitle(f'{name.capitalize()} in evaluation mode: {scope}')
        fig.tight_layout()
        temporary = output / f'{name}_curves.tmp.png'
        fig.savefig(temporary, dpi=150)
        temporary.replace(output / f'{name}_curves.png')
        plt.close(fig)
