"""
Write synthetic line items into the empty rows of REAL nota photographs.

Rendering a nota from scratch produces a page that only looks like a nota to a
renderer. Training on it taught the recogniser our fonts and our flat paper: it
moved held-out-layout accuracy 13 points and real-receipt accuracy by nothing.

Real photographs already carry everything the renderer fakes badly -- paper
grain, phone-camera noise, uneven light, shadows, hands, background, genuine
perspective. And their tables are mostly empty: 439 pages offer ~8,000 unused
row bands. So we keep the photograph and add only the ink.

    real page  ->  find the ruled table  ->  pick empty rows
               ->  draw a plausible line item with a handwriting face
               ->  multiply-blend so paper texture shows through the stroke
               ->  emit crops + labels

What this gives that the renderer cannot: real-world imaging.
What it cannot give: layout diversity -- every page is the same booklet. Use it
alongside rendered pages, not instead of them.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

# Column centres for this booklet, as fractions of width (docs/dataset-audit.html)
COLUMN_CENTRE = {"qty": 0.115, "nama": 0.40, "harga": 0.695, "jumlah": 0.86}
COLUMN_WIDTH = {"qty": 0.17, "nama": 0.38, "harga": 0.13, "jumlah": 0.17}

SOURCE_RE = re.compile(r"^(.*?)_jpg\.rf\.")


def source_id(name):
    m = SOURCE_RE.match(name)
    return m.group(1) if m else name


def ruled_bands(page: np.ndarray, x0_frac=0.10, x1_frac=0.95):
    """Find the printed horizontal rules, and return the gaps between them.

    Estimating row pitch from text height drifts out of sync with the printed
    rules and lands new writing on top of them, which trains the recogniser on
    half-occluded glyphs. The lines themselves are the ground truth.
    """
    H, W = page.shape[:2]
    strip = page[:, int(W * x0_frac):int(W * x1_frac)].mean(axis=2)
    darkness = (strip < strip.mean() - 25).sum(axis=1).astype(float)
    if darkness.max() < 1:
        return []
    span = int(W * (x1_frac - x0_frac))
    is_line = darkness > span * 0.55           # a rule spans most of the table

    rows, run = [], []
    for y, flag in enumerate(is_line):
        if flag:
            run.append(y)
        elif run:
            rows.append(sum(run) // len(run))
            run = []
    if run:
        rows.append(sum(run) // len(run))

    bands = []
    for a, b in zip(rows, rows[1:]):
        if 0.012 * H < (b - a) < 0.06 * H:     # plausible row height
            bands.append((a, b))
    return bands


def ink_colour(page: np.ndarray, boxes) -> tuple:
    """Sample the darkest pixels of existing handwriting so new ink matches."""
    samples = []
    for b in boxes[:6]:
        x, y, w, h = (int(v) for v in b["bbox"])
        patch = page[max(0, y):y + h, max(0, x):x + w].reshape(-1, 3)
        if len(patch):
            dark = patch[patch.sum(1).argsort()[: max(1, len(patch) // 10)]]
            samples.append(dark.mean(0))
    if not samples:
        return (35, 35, 45)
    return tuple(int(v) for v in np.mean(samples, axis=0))


def fonts_in(directory) -> list:
    return sorted(str(p) for p in Path(directory).glob("*.ttf")) if directory else []


def build(raw_dir, out_dir, font_dir, vocabulary, n_pages=400, seed=0,
          max_new_rows=4, save_pages=False):
    rng = random.Random(seed)
    raw, out = Path(raw_dir), Path(out_dir)
    (out / "crops").mkdir(parents=True, exist_ok=True)

    coco = json.loads((raw / "_annotations.coco.json").read_text())
    images = {i["id"]: i for i in coco["images"]}
    anns = collections.defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    faces = fonts_in(font_dir)
    if not faces:
        raise SystemExit("no handwriting fonts found in " + str(font_dir))

    seen, records, made = set(), [], 0
    for iid, boxes in sorted(anns.items()):
        if made >= n_pages:
            break
        name = images[iid]["file_name"]
        sid = source_id(name)
        if sid in seen or not boxes:
            continue
        seen.add(sid)

        path = raw / name
        if not path.exists():
            continue
        page = Image.open(path).convert("RGB")
        W, H = page.size
        arr = np.array(page)

        # Anchor on the last ITEM row, not the last annotation: the lowest
        # writing on the page is the "Jumlah Rp." total and the signature line,
        # and anchoring there leaves no room above the footer.
        body = [b for b in boxes
                if (b["bbox"][1] + b["bbox"][3] / 2) < H * 0.62]
        if not body:
            continue
        ys = sorted((b["bbox"][1] + b["bbox"][3] / 2) for b in body)
        hs = sorted(b["bbox"][3] for b in body)
        text_h = hs[len(hs) // 2]
        pitch = text_h * 2.1                      # ruled-row spacing on this form
        last = ys[-1]

        # Empty ruled bands below the existing writing, stopping short of the
        # total line. Centres come from the printed rules, not from arithmetic.
        bands = ruled_bands(arr)
        used = {round((b["bbox"][1] + b["bbox"][3] / 2)) for b in boxes}
        free = [(a, b) for a, b in bands
                if a > last + pitch * 0.4 and b < H * 0.70
                and not any(a <= u <= b for u in used)]
        rng.shuffle(free)
        candidates = [(a + b) / 2 for a, b in free[:max_new_rows]]
        band_h = min((b - a) for a, b in bands) if bands else pitch
        if not candidates:
            continue

        colour = ink_colour(arr, boxes)
        face_path = rng.choice(faces)
        draw = ImageDraw.Draw(page)

        for cy in candidates:
            nama, base = rng.choice(vocabulary)
            harga = max(500, int(base * rng.uniform(0.85, 1.2)) // 500 * 500)
            qty = rng.choice([1, 1, 2, 2, 3, 4, 5, 10, 12])
            # Plain digits, no thousand separators: the corpus labels are
            # bare numerals and the character dictionary has no ".".
            values = {"qty": str(qty), "nama": nama,
                      "harga": str(harga), "jumlah": str(qty * harga)}

            for role, text in values.items():
                cx = COLUMN_CENTRE[role] * W
                max_w = COLUMN_WIDTH[role] * W * 0.85
                size = int(min(text_h, band_h * 0.72) * rng.uniform(0.92, 1.08))
                while size > 8:
                    font = ImageFont.truetype(face_path, size)
                    if draw.textlength(text, font=font) <= max_w:
                        break
                    size -= 1
                font = ImageFont.truetype(face_path, max(size, 8))
                bbox = draw.textbbox((cx, cy), text, font=font, anchor="mm")

                # Draw the glyph on white, then multiply into the page so the
                # photograph's paper grain and shading show through the stroke.
                pad = 4
                x0, y0 = int(bbox[0]) - pad, int(bbox[1]) - pad
                x1, y1 = int(bbox[2]) + pad, int(bbox[3]) + pad
                if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                    continue
                stamp = Image.new("RGB", (x1 - x0, y1 - y0), "white")
                # Draw with the SAME anchor the box was measured with. Using the
                # default top-left here puts the glyph lower than the stamp and
                # silently shears off its bottom.
                ImageDraw.Draw(stamp).text((cx - x0, cy - y0), text, font=font,
                                           fill=colour, anchor="mm")
                if rng.random() < 0.5:
                    stamp = stamp.rotate(rng.uniform(-2.5, 2.5), expand=False,
                                         fillcolor="white", resample=Image.BICUBIC)
                region = page.crop((x0, y0, x1, y1))
                page.paste(ImageChops.multiply(region, stamp), (x0, y0))
                records.append({"source": sid, "role": role, "text": text,
                                "box": (x0, y0, x1 - x0, y1 - y0)})

        if save_pages:
            (out / 'pages').mkdir(exist_ok=True)
            page.save(out / 'pages' / f'{sid}.jpg', quality=92)
        composed = np.array(page)
        for r in [r for r in records if r["source"] == sid]:
            x, y, w, h = r["box"]
            crop = composed[y:y + h, x:x + w]
            if crop.shape[0] < 6 or crop.shape[1] < 6:
                continue
            fname = f"{sid}_{len(list((out/'crops').iterdir())):06d}.jpg"
            Image.fromarray(crop).save(out / "crops" / fname, quality=95)
            r["file"] = f"crops/{fname}"
        made += 1

    lines = [f"{r['file']}\t{r['text']}" for r in records if r.get("file")]
    (out / "rec.txt").write_text("\n".join(lines) + "\n")
    print(f"pages composited {made}")
    print(f"crops written    {len(lines)}")
    return len(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--catalogue", default=None)
    ap.add_argument("--pages", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from synth_nota import load_vocabulary
    build(args.raw, args.out, args.fonts,
          load_vocabulary(args.catalogue), args.pages, args.seed)
