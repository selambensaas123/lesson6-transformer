"""Train on a Gutenberg book, validate on held-out paragraphs, and export metrics."""
import json
from pathlib import Path
import re

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from generator import BOS, EOS, PAD, GeneratorTransformer, encode

ROOT = Path(__file__).resolve().parent


class Windows(Dataset):
    """Cover every paragraph transition once with shifted non-overlapping windows."""

    def __init__(self, paragraphs, length):
        self.pairs = []
        for paragraph in paragraphs:
            ids = [BOS, *encode(paragraph), EOS]
            for start in range(0, len(ids) - 1, length):
                x = ids[start:start + length]
                y = ids[start + 1:start + length + 1]
                x = x[:len(y)]
                extra = [PAD] * (length - len(y))
                self.pairs.append((torch.tensor(x + extra), torch.tensor(y + extra)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


def paragraphs(path):
    """Remove Gutenberg metadata and normalize paragraph whitespace."""
    content = path.read_text(encoding='utf-8-sig')
    start = re.search(r'^\*\*\* START .*$', content, flags=re.M)
    end = re.search(r'^\*\*\* END .*$', content, flags=re.M)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError('Expected Gutenberg start and end markers')
    return [' '.join(block.split()) for block in re.split(r'\n\s*\n', content[start.end():end.start()])
            if block.strip()]


def run_epoch(model, loader, optimizer=None):
    """Return byte-token average cross entropy and accuracy."""
    model.train(optimizer is not None)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            logits = model(x)
            tokens = y.ne(PAD).sum().item()
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=PAD,
                                   reduction='sum')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / tokens).backward()
                optimizer.step()
            total_loss += loss.item()
            count += tokens
            correct += ((logits.argmax(-1) == y) & y.ne(PAD)).sum().item()
    return dict(loss=total_loss / count, accuracy=correct / count, tokens=count)


def main():
    """Train for three epochs and record actual sampling and beam examples."""
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = GeneratorTransformer()
    blocks = paragraphs(ROOT / 'data' / 'alice.txt')
    val = Windows(blocks[9::10], 128)
    train = Windows([p for i, p in enumerate(blocks) if i % 10 != 9], 128)
    rng = torch.Generator().manual_seed(42)
    training = DataLoader(train, batch_size=8, shuffle=True, generator=rng)
    validation = DataLoader(val, batch_size=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = []
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    for epoch in range(1, 4):
        row = dict(epoch=epoch, train=run_epoch(model, training, optimizer),
                   val=run_epoch(model, validation))
        row['val']['perplexity'] = math.exp(row['val']['loss'])
        history.append(row)
        print(row, flush=True)
    torch.save(dict(config=model.config, state=model.state_dict()), output / 'checkpoint.pt')
    examples = {}
    for prompt in ('Alice was', 'The Rabbit'):
        examples[prompt] = dict(sample=model.generate(prompt, max_out_tokens=100, temperature=.8),
                                beam=model.generate(prompt, max_out_tokens=100,
                                                    temperature=.8, beam_width=3))
    (output / 'metrics.json').write_text(json.dumps(dict(history=history, examples=examples),
                                                    indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'val'):
            axis.plot([r['epoch'] for r in history], [r[split][metric] for r in history],
                      marker='o', label=split)
        axis.set(xlabel='Epoch', ylabel=metric, xticks=[1, 2, 3])
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'learning.png', dpi=140)


if __name__ == '__main__':
    main()
