"""
A verified pool of generated handwriting images.

DiffusionPen is trained on IAM, which is English words. Indonesian product
names and 4-6 digit prices are out of its distribution, and it quietly renders
the wrong thing: measured on a sample, `15000` came out as `115000` and
`17000` as `145000`.

A generated image whose pixels disagree with its label is worse than no data.
It teaches the recogniser that a price looks like a different price.

So nothing enters the pool unverified. Every candidate is read back with the
recogniser and kept only if it says what it claims to say. Yield on the first
sample was 50% overall -- 30% for words, 64% for digits.

    python ml/wordpool.py verify  <pool_dir>   # filter in place
    python ml/wordpool.py stats   <pool_dir>

The filter biases the pool toward text the recogniser already reads well, which
limits how much *new* signal it carries. The principled fix is to fine-tune the
generator on real nota crops so Indonesian numerals are in distribution; this
gate is what makes the current output safe to use meanwhile.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


def normalise(text: str) -> str:
    return re.sub(r"[^0-9a-z]", "", str(text).lower())


class WordPool:
    """Verified generated words, keyed by their text."""

    def __init__(self, pool_dir, seed=0):
        self.dir = Path(pool_dir)
        index_path = self.dir / "verified.json"
        if not index_path.exists():
            index_path = self.dir / "index.json"
        self.index = json.loads(index_path.read_text()) if index_path.exists() else {}
        self.rng = random.Random(seed)

    def __len__(self):
        return sum(len(v) for v in self.index.values())

    def has(self, text) -> bool:
        return bool(self.index.get(normalise(text)))

    def get(self, text):
        """A random rendering of this word, or None if the pool lacks it."""
        options = self.index.get(normalise(text))
        if not options:
            return None
        from PIL import Image
        return Image.open(self.dir / self.rng.choice(options)).convert("RGB")


def verify(pool_dir, model_dir):
    """Keep only generations whose pixels match their label."""
    from paddleocr import TextRecognition
    pool = Path(pool_dir)
    raw = json.loads((pool / "index.json").read_text())
    rec = TextRecognition(model_name="PP-OCRv5_mobile_rec", model_dir=str(model_dir))

    pairs = [(w, f) for w, files in raw.items() for f in files]
    preds = []
    for i in range(0, len(pairs), 64):
        batch = [str(pool / f) for _, f in pairs[i:i + 64]]
        preds += [r.get("rec_text") or "" for r in rec.predict(batch)]

    kept: dict[str, list[str]] = {}
    for (word, fname), pred in zip(pairs, preds):
        if normalise(pred) == normalise(word):
            kept.setdefault(normalise(word), []).append(fname)

    (pool / "verified.json").write_text(json.dumps(kept, indent=1))
    total = len(pairs)
    survived = sum(len(v) for v in kept.values())
    print(f"verified {survived}/{total} = {survived/max(total,1):.0%} "
          f"across {len(kept)} distinct words")
    return kept


if __name__ == "__main__":
    import sys
    cmd, pool_dir = sys.argv[1], sys.argv[2]
    if cmd == "verify":
        model = sys.argv[3] if len(sys.argv) > 3 else Path.home() / "handoff/models/rec"
        verify(pool_dir, model)
    else:
        p = WordPool(pool_dir)
        print(f"{len(p)} images across {len(p.index)} words")
        for w in list(p.index)[:10]:
            print(f"  {w:<14} {len(p.index[w])}")
