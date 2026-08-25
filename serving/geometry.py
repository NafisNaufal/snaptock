"""
Put the page the right way up, and cut each text line out squarely.

A photographed nota is rarely upright. The paper is held in a hand, shot
sideways, or turned a full 90 degrees on the desk. The recognizer is trained on
horizontal crops, so on a sideways page every crop is a tall vertical sliver,
every prediction comes back empty, and the parser downstream sees a blank
document. That is what a rotated photo actually looked like in production.

Nothing here is trained. The detector already returns one quadrilateral per text
line, and text lines run along the page's writing direction -- so the average
edge angle of those quads *is* the page angle. That handles tilt and all four
quarter turns. It cannot tell upright from upside down, because a line and the
same line rotated 180 degrees lie at the same angle; that last bit is settled by
asking the recognizer which way round it reads with more confidence.

The second half of the file is the crop itself. Taking a detected line's
axis-aligned bounding rectangle is only correct when the line is level: on a
hand-held page a slanted line's rectangle also swallows the paper and the
neighbouring lines around it. Warping the quadrilateral onto a rectangle instead
gives the recognizer the strip it was trained on.
"""

from __future__ import annotations

import cv2
import numpy as np

MIN_LINES = 4          # below this the angle estimate is noise, so leave the page alone
MIN_TILT = 0.7         # degrees; under this a rotation costs resampling and buys nothing
FLIP_SAMPLE = 8        # longest lines used to decide upright vs upside down


def _long_edge(quad: np.ndarray) -> tuple[float, float]:
    """Direction of a text line: its longer side, as (angle in radians, length)."""
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    dx, dy = max(edges, key=lambda e: e[0] ** 2 + e[1] ** 2)
    return float(np.arctan2(dy, dx)), float(np.hypot(dx, dy))


def page_angle(polys) -> float:
    """
    Length-weighted average text angle in degrees, in [-90, 90).

    Angles are averaged doubled and then halved. A line direction is only
    defined modulo 180 degrees, so a page of vertical lines reads as -90 on some
    quads and +90 on others; plain averaging would cancel them to zero and leave
    the page sideways. Doubling maps both onto the same direction first.
    """
    vectors = []
    for poly in polys:
        angle, length = _long_edge(np.asarray(poly, dtype=np.float64))
        vectors.append((length * np.cos(2 * angle), length * np.sin(2 * angle)))
    if not vectors:
        return 0.0
    cos, sin = np.sum(vectors, axis=0)
    return float(np.degrees(np.arctan2(sin, cos)) / 2)


def rotate(page: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate counter-clockwise about the centre, growing the canvas to fit."""
    if abs(degrees) < MIN_TILT:
        return page
    if abs(degrees - round(degrees / 90) * 90) < MIN_TILT:
        return np.rot90(page, k=int(round(degrees / 90)) % 4).copy()

    h, w = page.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w, out_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += out_w / 2 - w / 2
    matrix[1, 2] += out_h / 2 - h / 2
    return cv2.warpAffine(page, matrix, (out_w, out_h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew(page: np.ndarray, polys, detect, passes: int = 3):
    """
    Level the page, then detect again on it, and repeat until it stops moving.

    Re-detecting is not optional: the old polygons index into the old canvas,
    and rotating them by hand would keep the detector's mistakes on the sideways
    image, which is where it is weakest.

    Nor is one pass enough. On a page lying at 86 degrees the detector is
    working at its worst, and the angle it gives back is a couple of degrees
    out; only once the page is roughly level does it report the rest. Two
    degrees sounds harmless and is not -- across a 3000-pixel page it drags one
    end of every row a full row-pitch away from the other, and the table
    collapses into a single row. Each pass therefore refines the running angle
    and rotates the *original* again, so the pixels are only resampled once.
    """
    original, total = page, 0.0
    for _ in range(passes):
        if len(polys) < MIN_LINES:
            break
        angle = page_angle(polys)
        if abs(angle) < MIN_TILT:
            break
        turned = rotate(original, total + angle)
        found = detect(turned)
        if len(found) < MIN_LINES:      # rotation lost us the page; keep what worked
            break
        page, polys, total = turned, found, total + angle
    return page, polys, total


def upright(page: np.ndarray, polys, recognize):
    """
    Decide whether a levelled page is also the right way up.

    Reads the longest few lines both ways and keeps the orientation the
    recognizer is more sure of. Long lines are the item names, which carry real
    words -- a confidence gap on those means something, where a gap on a
    two-digit quantity does not.
    """
    if len(polys) < MIN_LINES:
        return page, polys

    longest = sorted(polys, key=lambda p: cv2.contourArea(
        np.asarray(p, dtype=np.float32)), reverse=True)[:FLIP_SAMPLE]
    crops = [c for c in (rectify(page, p) for p in longest) if c is not None]
    if not crops:
        return page, polys

    as_is = _mean_score(recognize(crops))
    flipped = _mean_score(recognize([np.rot90(c, 2).copy() for c in crops]))
    if flipped <= as_is:
        return page, polys

    turned = np.rot90(page, 2).copy()
    return turned, turn_boxes(turned, polys)


def turn_boxes(page, polys):
    """A 180 degree turn is exact, so the boxes can be carried over arithmetically."""
    h, w = page.shape[:2]
    return [np.asarray([[w - x, h - y] for x, y in poly])[::-1] for poly in polys]


def _mean_score(predictions) -> float:
    scores = [float(p.get("rec_score") or 0.0) for p in predictions]
    return sum(scores) / len(scores) if scores else 0.0


def _order(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    total, diff = quad.sum(axis=1), np.diff(quad, axis=1).ravel()
    return np.array([quad[np.argmin(total)], quad[np.argmin(diff)],
                     quad[np.argmax(total)], quad[np.argmax(diff)]],
                    dtype=np.float32)


def rectify(page: np.ndarray, poly, padding: float = 0.05):
    """
    Warp one detected quadrilateral onto an upright rectangle.

    Padding matches the margin the training crops were cut with, so the strip
    the recognizer sees here looks like the strips it was fine-tuned on.
    """
    quad = _order(np.asarray(poly, dtype=np.float32))
    quad = quad.mean(axis=0) + (quad - quad.mean(axis=0)) * (1 + padding)

    width = int(round(max(np.linalg.norm(quad[1] - quad[0]),
                          np.linalg.norm(quad[2] - quad[3]))))
    height = int(round(max(np.linalg.norm(quad[3] - quad[0]),
                           np.linalg.norm(quad[2] - quad[1]))))
    if width < 4 or height < 4:
        return None

    target = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                      dtype=np.float32)
    patch = cv2.warpPerspective(page, cv2.getPerspectiveTransform(quad, target),
                                (width, height), borderMode=cv2.BORDER_REPLICATE)
    # A line the detector found standing on end: stand it back up.
    return np.rot90(patch).copy() if height >= width * 1.5 else patch


def bounds(poly) -> dict:
    """Axis-aligned extent, which is all the table parser needs of a line."""
    quad = np.asarray(poly, dtype=np.float64)
    x, y = quad[:, 0].min(), quad[:, 1].min()
    return {"x": float(x), "y": float(y),
            "w": float(quad[:, 0].max() - x), "h": float(quad[:, 1].max() - y)}
