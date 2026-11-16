"""Decoder-only language model using a fixed UTF-8 byte vocabulary."""
import torch
from torch import nn
from torch.nn import functional as F

from transformer_basics.layers import PositionalEncoding

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259


def encode(text):
    """Map UTF-8 bytes to IDs without fitting on any split."""
    return list(text.encode('utf-8'))


def decode(ids):
    """Decode byte IDs, omitting control tokens."""
    return bytes(i for i in ids if 0 <= i < 256).decode('utf-8', errors='replace')


class DecoderBlock(nn.Module):
    """Pre-normalized causal self-attention and a residual MLP."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                          nn.Linear(width * 4, width))

    def forward(self, x, forbidden):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, attn_mask=forbidden, need_weights=False)[0]
        return x + self.feed_forward(self.norm2(x))


class GeneratorTransformer(nn.Module):
    """Predict next bytes using only present and earlier positions."""

    def __init__(self, max_length=128, width=64, heads=4, blocks=2):
        super().__init__()
        self.config = dict(max_length=max_length, width=width, heads=heads, blocks=blocks)
        self.embedding = nn.Embedding(VOCAB, width, padding_idx=PAD)
        self.position = PositionalEncoding(width, max_length)
        self.blocks = nn.ModuleList(DecoderBlock(width, heads) for _ in range(blocks))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, VOCAB))
        self.register_buffer('causal', torch.ones(max_length, max_length, dtype=torch.bool).triu(1),
                             persistent=False)

    def forward(self, ids):
        length = ids.shape[1]
        x = self.position(self.embedding(ids))
        for block in self.blocks:
            x = block(x, self.causal[:length, :length])
        return self.output(x)

    def _options(self, ids, limit, temperature):
        x = torch.tensor([ids[-limit:]], device=self.embedding.weight.device)
        logits = self(x)[0, -1] / temperature
        logits[[PAD, BOS]] = -torch.inf
        return logits

    @torch.no_grad()
    def generate(self, prompt, context_len=50, temperature=1., max_out_tokens=200,
                 beam_width=1):
        """Generate by sampling or beam search; stop each sequence at EOS."""
        if not 1 <= context_len <= self.config['max_length']:
            raise ValueError('Context length exceeds the trained window')
        if temperature <= 0 or beam_width < 1 or beam_width > 257 or max_out_tokens < 0:
            raise ValueError('Invalid generation parameters')
        self.eval()
        candidates = [([BOS, *encode(prompt)], 0.)]
        for _ in range(max_out_tokens):
            extended = []
            for ids, score in candidates:
                if ids[-1] == EOS:
                    extended.append((ids, score))
                    continue
                logits = self._options(ids, context_len, temperature)
                log_probs = F.log_softmax(logits, dim=-1)
                if beam_width == 1:
                    tokens = [torch.multinomial(log_probs.exp(), 1).item()]
                else:
                    tokens = log_probs.topk(beam_width).indices.tolist()
                extended.extend(([*ids, token], score + log_probs[token].item())
                                for token in tokens)
            candidates = sorted(extended, key=lambda item: item[1], reverse=True)[:beam_width]
            if all(ids[-1] == EOS for ids, _ in candidates):
                break
        return decode(candidates[0][0])

    @classmethod
    def load_from_checkpoint(cls, path):
        """Restore model dimensions and learned weights on CPU."""
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(**saved['config'])
        model.load_state_dict(saved['state'])
        return model.eval()
