"""Regression tests for causal masking, transitions, UTF-8 and generation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from generator import BOS, EOS, PAD, VOCAB, GeneratorTransformer, encode, decode
from train_generator import Windows


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_utf8_roundtrip(self):
        self.assertEqual(decode([BOS, *encode('Çay и чай'), EOS]), 'Çay и чай')

    def test_windows_cover_every_transition(self):
        dataset = Windows(['abcdef'], 3)
        expected = [BOS, *encode('abcdef'), EOS]
        actual = [(a.item(), b.item()) for x, y in dataset for a, b in zip(x, y) if b != PAD]
        self.assertEqual(actual, list(zip(expected, expected[1:])))

    def test_future_cannot_change_earlier_logits(self):
        model = GeneratorTransformer(max_length=8, width=16, blocks=1).eval()
        with torch.no_grad():
            a = model(torch.tensor([[BOS, 1, 2, 3]]))
            b = model(torch.tensor([[BOS, 1, 20, 30]]))
        torch.testing.assert_close(a[:, :2], b[:, :2])

    def test_sliding_window_and_eos(self):
        model = GeneratorTransformer(max_length=4, width=16, blocks=1)
        observed = []
        def options(ids, limit, temperature):
            observed.append(ids[-limit:])
            logits = torch.full((VOCAB,), -torch.inf)
            logits[ord('d') if len(observed) == 1 else EOS] = 0.
            return logits
        with patch.object(model, '_options', side_effect=options):
            self.assertEqual(model.generate('abc', context_len=3), 'abcd')
        self.assertEqual(observed, [encode('abc'), encode('bcd')])

    def test_checkpoint_roundtrip(self):
        model = GeneratorTransformer(max_length=4, width=16, blocks=1).eval()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'
            torch.save(dict(config=model.config, state=model.state_dict()), path)
            loaded = GeneratorTransformer.load_from_checkpoint(path)
        x = torch.tensor([[BOS, 65]])
        torch.testing.assert_close(model(x), loaded(x))


if __name__ == '__main__':
    unittest.main()
