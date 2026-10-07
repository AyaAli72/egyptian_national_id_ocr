"""Finds the first-name / father's-name / address bands on an Egyptian ID card,
with no model: anchor on the national-number row (digit_reader), then detect the
black text lines above it and group them by order.

    python detect_fields.py card.jpg [more.jpg ...] [--out debug_fields]

Saves <name>_fields.png (boxes drawn on the card) and one crop per field.
"""
import os
import sys
import numpy as np
import cv2

import digit_reader

# (ink must be this much darker than the local paper, and darker than this in absolute terms), strict -> loose:
# sharp scans need the strict one (the loose one merges lines into the artwork), webcam shots need the loose one
INK_SETTINGS = ((60, 120), (45, 150), (35, 255))
FIELD_COLORS = {"first_name": (0, 0, 255), "father_name": (0, 170, 0), "address": (255, 0, 0)}


def _frame(img):
    """The reader's working frame: image shrunk to <=1400px and (if that is what
    found the number) rotated. Returns (gray, color, number_boxes)."""
    cand = next(digit_reader.segment_candidates(img), None)
    if cand is None:
        return None
    h, w = img.shape[:2]
    scale = 1400.0 / max(h, w) if max(h, w) > 1400 else 1.0
    color = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
    angle = cand["params"][0]
    if angle:
        M = cv2.getRotationMatrix2D((color.shape[1] / 2, color.shape[0] / 2), angle, 1.0)
        color = cv2.warpAffine(color, M, (color.shape[1], color.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return cand["gray"], color, cand["boxes"]


def _ink_mask(gray, diff, dark):
    """Text ink: clearly darker than the paper around it AND dark in absolute terms."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    paper = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    return (((paper.astype(np.int16) - g) > diff) & (g < dark)).astype(np.uint8)


def find_lines(gray, number_boxes, diff, dark):
    """Text lines above the number row, right of the photo. Returns [(x, y, w, h)] top to bottom."""
    H, W = gray.shape
    num_left = min(b[0] for b in number_boxes)
    num_right = max(b[0] + b[2] for b in number_boxes)
    num_top = min(b[1] for b in number_boxes)
    mh = float(np.median([b[3] for b in number_boxes]))
    x0, x1 = max(0, num_left - int(0.10 * W)), min(W, num_right + int(0.03 * W))
    y1 = max(1, num_top - int(0.3 * mh))
    ink = _ink_mask(gray, diff, dark)[:y1, x0:x1]
    rows = np.convolve(ink.sum(axis=1).astype(np.float32), np.ones(3) / 3, mode="same")
    on = rows > max(2.0, 0.012 * (x1 - x0))
    bands, start = [], None
    for y, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append([start, y])
            start = None
    merged = []
    for b in bands:                                   # dots above/below letters split a line: rejoin small gaps
        if merged and b[0] - merged[-1][1] <= 0.3 * mh:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    lines = []
    for ya, yb in merged:
        if yb - ya < 0.5 * mh:
            continue
        cols = np.where(ink[ya:yb].sum(axis=0) > 0)[0]
        if len(cols) == 0 or cols[-1] - cols[0] < 1.0 * mh:
            continue
        lines.append((x0 + int(cols[0]), ya, int(cols[-1] - cols[0] + 1), yb - ya))
    return lines


def text_block(lines):
    """The evenly spaced block of lines closest to the number row (4, or 5 if a
    name wraps). Whatever sits above it with a much bigger gap is the card header
    or junk and is dropped. Returns the block, or None."""
    if len(lines) < 4:
        return None
    block = list(lines[-4:])
    pitch = float(np.median(np.diff([l[1] for l in block])))
    i = len(lines) - 5
    while i >= 0 and len(block) < 5 and lines[i + 1][1] - lines[i][1] <= 1.6 * pitch:
        block.insert(0, lines[i])
        i -= 1
    hs = np.array([l[3] for l in block], float)
    if hs.max() > 2.2 * np.median(hs):                 # one "line" is far taller than the rest: bands got merged
        return None
    gaps = np.diff([l[1] for l in block])
    if gaps[0] > 1.6 * np.median(gaps[1:]):            # a big gap at the top: the header slipped in as a line
        return None
    return block


def group_fields(block):
    """first name = top line. The rest split into father's name | address at the
    LARGEST vertical gap (on every card so far the gap before the address is the
    biggest) -- a layout heuristic, not a guarantee; the address's last line should
    contain a governorate once the text is read."""
    rest = block[1:]
    gaps = [rest[i + 1][1] - rest[i][1] for i in range(len(rest) - 1)]
    k = int(np.argmax(gaps)) + 1
    return {"first_name": block[:1], "father_name": rest[:k], "address": rest[k:]}


# extra room around each band, in digit-heights of the number row: the dots above/below
# Arabic letters sit outside the tight line box, and a one-word first name needs context
PAD = {"first_name": (0.8, 0.7), "father_name": (0.4, 0.4), "address": (0.4, 0.4)}   # (sideways, up/down)


def _union(lines, pad_x, pad_y, shape=None):
    x0 = min(l[0] for l in lines) - pad_x
    y0 = min(l[1] for l in lines) - pad_y
    x1 = max(l[0] + l[2] for l in lines) + pad_x
    y1 = max(l[1] + l[3] for l in lines) + pad_y
    if shape:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)
    return x0, y0, x1 - x0, y1 - y0


def detect(img):
    """{'fields': {name: (x, y, w, h)}, 'lines': [...], 'problem': str|None, 'color': frame}"""
    fr = _frame(img)
    if fr is None:
        return {"fields": {}, "lines": [], "problem": "number row not found", "color": img}
    gray, color, boxes = fr
    lines, block = [], None
    for diff, dark in INK_SETTINGS:
        lines = find_lines(gray, boxes, diff, dark)
        block = text_block(lines)
        if block:
            break
    if not block:
        return {"fields": {}, "lines": lines, "problem": f"could not find 4-5 evenly spaced text lines above the number ({len(lines)} found)",
                "color": color, "number": boxes}
    mh = float(np.median([b[3] for b in boxes]))
    fields = {k: _union(v, int(PAD[k][0] * mh), int(PAD[k][1] * mh), shape=gray.shape)
              for k, v in group_fields(block).items()}
    return {"fields": fields, "lines": lines, "problem": None, "color": color, "number": boxes}


def main(paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read")
            continue
        r = detect(img)
        name = os.path.splitext(os.path.basename(path))[0]
        dbg = r["color"].copy()
        for (x, y, w, h) in r["lines"]:
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (150, 150, 150), 1)
        for k, (x, y, w, h) in r["fields"].items():
            cv2.rectangle(dbg, (x, y), (x + w, y + h), FIELD_COLORS[k], 2)
            cv2.putText(dbg, k, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FIELD_COLORS[k], 1)
            cv2.imwrite(os.path.join(out_dir, f"{name}_{k}.png"), r["color"][y:y + h, x:x + w])
        for (x, y, w, h) in r.get("number", []):
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 200, 200), 1)
        cv2.imwrite(os.path.join(out_dir, f"{name}_fields.png"), dbg)
        print(f"{path}: {len(r['lines'])} lines -> " + (f"FLAG: {r['problem']}" if r["problem"] else "OK"))


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "debug_fields"
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    if not args:
        print(__doc__)
        sys.exit(1)
    main(args, out)
