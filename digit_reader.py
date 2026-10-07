"""Reads the 14-digit Egyptian national ID number WITHOUT a vision-language model.

Why: general OCR engines and small VLMs are known to fail on this number (they
drop the zeros - printed as dots - and return 10-11 digits; see EasyOCR issue
#1453). The reliable approach is the classic one: find the number's line,
cut it into one image per character (digits AND dots), classify each
character with a small classifier trained on real crops of this card font,
then use the number's own structure (century, birth date, governorate) to
reject impossible readings and pick the best valid one.

The classifier is a nearest-neighbour search over a bank of real glyph crops
(digit_templates/<digit>/*.png, all taken from real cards), expanded with blur
/ rotation / scale / noise augmentation so webcam and phone photos both match.
Numpy + OpenCV only. When an operator confirms or corrects a number, the
glyphs are saved back into digit_templates/ so the reader improves with use.
"""
import os
import glob
import hashlib
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(HERE, "digit_templates")
EXPECTED = 14



# --------------------------------------------------------------------------
# Segmentation: the number's line -> 14 character boxes
# --------------------------------------------------------------------------

def _binarize(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return g, cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 14)


def _lines_from(gray, bs, C):
    """Candidate text lines of digit-sized glyphs in a grey image."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, C)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    comps = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in st[i]]
        if area < 10 or bh < 0.006 * H or bh > 0.16 * H or bw > 1.4 * bh:
            continue
        # ink = clearly darker than the paper right around it (a LOCAL test: the card, a hand and the
        # background all have different brightness, so a global threshold fails on webcam frames)
        pad = max(4, bh // 2)
        y0, y1, x0, x1 = max(0, y - pad), min(H, y + bh + pad), max(0, x - pad), min(W, x + bw + pad)
        window = g[y0:y1, x0:x1]
        paper = float(np.percentile(window, 80))
        ink = float(g[y:y + bh, x:x + bw][lab[y:y + bh, x:x + bw] == i].mean())
        if paper - ink < 35 or ink > 0.8 * paper:
            continue
        comps.append((x, y, bw, bh, area))
    glyphs = [c for c in comps if c[3] >= 0.012 * H and c[2] >= 0.2 * c[3] and c[4] / float(c[2] * c[3]) >= 0.22]
    glyphs.sort(key=lambda m: m[1] + m[3])
    groups, cur = [], []
    for m in glyphs:                                       # text lines, grouped by baseline (dots sit low, tall digits high)
        if cur and abs((m[1] + m[3]) - np.mean([c[1] + c[3] for c in cur])) > 0.6 * np.median([c[3] for c in cur]):
            groups.append(cur)
            cur = []
        cur.append(m)
    if cur:
        groups.append(cur)
    lines = []
    for r in groups:
        r = sorted(r, key=lambda m: m[0])
        w_med = float(np.median([m[2] for m in r]))
        run = [r[0]]
        for m in r[1:] + [None]:                           # split on big gaps (e.g. the issue date printed on the left of the same baseline)
            if m is not None and m[0] - (run[-1][0] + run[-1][2]) <= 4.5 * w_med:
                run.append(m)
                continue
            if len(run) >= 8:
                hs = np.array([c[3] for c in run], float)
                if hs.std() / hs.mean() <= 0.4:
                    lines.append(run)
            run = [m] if m is not None else []
    return comps, lines


def _chars_of_line(line, comps):
    """Digits of a line + the dots (zeros) around them -> exactly 14 boxes, or None."""
    med_h = float(np.median([m[3] for m in line]))
    med_w = float(np.median([m[2] for m in line]))
    base = float(np.median([m[1] + m[3] for m in line]))
    x0, x1 = min(m[0] for m in line), max(m[0] + m[2] for m in line)
    chars = list(line)
    for c in comps:                                        # zeros are small, roundish, dark dots on the baseline
        if c in chars:
            continue
        x, y, bw, bh, area = c
        if bh <= 0.6 * med_h and bw <= 0.9 * med_w and area / float(bw * bh) >= 0.35 \
                and abs((y + bh) - base) <= 0.45 * med_h and x0 - 2.2 * med_w <= x <= x1 + 2.2 * med_w:
            chars.append(c)
    chars.sort(key=lambda m: m[0])
    merged = []
    for m in chars:                                        # merge parts of one glyph that were split
        if merged and m[0] < merged[-1][0] + merged[-1][2] * 0.5 and not (m[3] <= 0.6 * med_h and merged[-1][3] <= 0.6 * med_h):
            p = merged[-1]
            nx0, ny0 = min(p[0], m[0]), min(p[1], m[1])
            nx1, ny1 = max(p[0] + p[2], m[0] + m[2]), max(p[1] + p[3], m[1] + m[3])
            merged[-1] = (nx0, ny0, nx1 - nx0, ny1 - ny0, p[4] + m[4])
        else:
            merged.append(m)
    merged = _fill_gaps(merged, med_h)
    return merged if len(merged) == EXPECTED else None


ANGLES = (0, -4, 4, -8, 8)
THRESHOLDS = ((41, 14), (31, 9), (61, 18))


def segment_candidates(img, hint=None):
    """Yield every way of reading the image as 'a line of 14 characters':
    dicts {gray, boxes, virtual, params}. `hint` (params of an earlier frame)
    is tried first. Tilted cards are handled by trying small rotations."""
    h, w = img.shape[:2]
    scale = 1400.0 / max(h, w) if max(h, w) > 1400 else 1.0           # keep it fast on big frames
    base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    order = [(a, t) for a in ANGLES for t in THRESHOLDS]
    if hint in order:
        order.remove(hint)
        order.insert(0, hint)
    for angle, (bs, C) in order:
        gray = base
        if angle:
            M = cv2.getRotationMatrix2D((base.shape[1] / 2, base.shape[0] / 2), angle, 1.0)
            gray = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        comps, lines = _lines_from(gray, bs, C)
        for line in lines:
            chars = _chars_of_line(line, comps)
            if chars:
                yield {"gray": gray, "boxes": [m[:4] for m in chars], "virtual": [m[4] == 0 for m in chars],
                       "params": (angle, (bs, C))}


def debug_image(img, path):
    """Annotated picture of what the segmenter saw (components in grey, number-line
    candidates in colour) - saved when the number line could not be found."""
    h, w = img.shape[:2]
    scale = 1400.0 / max(h, w) if max(h, w) > 1400 else 1.0
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    comps, lines = _lines_from(gray, 41, 14)
    for (x, y, bw, bh, a) in comps:
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (160, 160, 160), 1)
    for i, line in enumerate(lines):
        col = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 200)][i % 4]
        for (x, y, bw, bh, a) in line:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), col, 2)
    cv2.putText(out, f"{len(lines)} candidate line(s) with 8+ digit-sized glyphs", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imencode('.png', out)[1].tofile(path)


def segment_number(img):
    """First candidate as (gray, boxes, virtual), or None (see segment_candidates)."""
    for cand in segment_candidates(img):
        return cand["gray"], cand["boxes"], cand["virtual"]
    return None


def _fill_gaps(chars, med_h):
    """A faint dot (a zero) can be too weak to be found. If two neighbours are
    ~2 pitches apart, there is a missing character between them: insert a dot."""
    if len(chars) < 8 or len(chars) >= EXPECTED:
        return chars
    cx = [m[0] + m[2] / 2 for m in chars]
    d = np.diff(cx)
    pitch = float(np.median(d))
    out = [chars[0]]
    for i in range(1, len(chars)):
        miss = int(round(d[i - 1] / pitch)) - 1
        if miss >= 1 and d[i - 1] > 1.6 * pitch:
            base = chars[i - 1][1] + chars[i - 1][3]
            for k in range(1, miss + 1):
                x = int(cx[i - 1] + d[i - 1] * k / (miss + 1))
                s_ = max(3, int(0.3 * med_h))
                out.append((x - s_ // 2, base - s_, s_, s_, 0))          # area 0 marks an inferred (virtual) dot
        out.append(chars[i])
    return out


# --------------------------------------------------------------------------
# Glyph images and the template bank
# --------------------------------------------------------------------------

def glyph32(gray, box, pad=0.25):
    """32x32 image of one character, ink bright on dark (the usual convention)."""
    x, y, w, h = [int(v) for v in box]
    s = max(8, int(max(w, h) * (1 + 2 * pad)))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - s // 2, cy - s // 2
    H, W = gray.shape
    crop = np.full((s, s), int(np.median(gray)), np.uint8)
    xs0, ys0, xs1, ys1 = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    crop[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = gray[ys0:ys1, xs0:xs1]
    return cv2.resize(255 - crop, (32, 32), interpolation=cv2.INTER_AREA)


def _unit(v):
    v = v.ravel().astype(np.float32)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def _features(g32):
    """Blurred 16x16 appearance + 16x16 gradient-magnitude map. (Measured
    better than HOG on held-out card photos, and needs nothing beyond
    OpenCV basics, which every OpenCV version has.)"""
    im = g32.astype(np.float32)
    lo, hi = np.percentile(im, 2), np.percentile(im, 98)
    im = np.clip((im - lo) / (hi - lo + 1e-3), 0, 1)
    soft = cv2.resize(cv2.GaussianBlur(im, (0, 0), 1.5), (16, 16), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(im, (0, 0), 1.0)
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0), cv2.Sobel(sm, cv2.CV_32F, 0, 1)
    grad = cv2.resize(np.sqrt(gx * gx + gy * gy), (16, 16), interpolation=cv2.INTER_AREA)
    return np.concatenate([_unit(soft), _unit(grad)])


def _augment(g32, rng, n):
    out = []
    for _ in range(n):
        a = g32.astype(np.float32)
        ang, sc = rng.uniform(-9, 9), rng.uniform(0.88, 1.12)
        M = cv2.getRotationMatrix2D((16, 16), ang, sc)
        M[:, 2] += rng.uniform(-1.5, 1.5, 2)
        a = cv2.warpAffine(a, M, (32, 32), borderMode=cv2.BORDER_REPLICATE)
        sig = rng.uniform(0, 1.8)
        if sig > 0.3:
            a = cv2.GaussianBlur(a, (0, 0), sig)
        a = a * rng.uniform(0.6, 1.2) + rng.uniform(-25, 25)
        a = a + rng.normal(0, rng.uniform(0, 8), a.shape)
        out.append(np.clip(a, 0, 255).astype(np.uint8))
    return out


_BANK = {}
_BUILTIN_LABELS = "30409211600357834092116003578"      # filled in below by tools/build_bank
_BUILTIN_B64 = "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"


def _templates():
    """All (digit, 32x32 image) templates: the built-in real-card crops plus
    any the operator has confirmed (digit_templates/<digit>/*.png)."""
    out = []
    if _BUILTIN_B64:
        import base64
        import zlib
        raw = np.frombuffer(zlib.decompress(base64.b64decode(_BUILTIN_B64)), np.uint8)
        labels = [int(c) for c in _BUILTIN_LABELS]
        for d, im in zip(labels, raw.reshape(-1, 32, 32)):
            out.append((d, im))
    for f in sorted(glob.glob(os.path.join(TEMPLATE_DIR, "processed", "[0-9]_*.png"))):
        g = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_GRAYSCALE)   # black glyph on white
        if g is None:
            continue
        ys, xs = np.where(g < 128)
        if len(xs) == 0:
            continue
        box = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        out.append((int(os.path.basename(f)[0]), glyph32(g, box)))         # same framing as query glyphs
    return out


def _load_bank(per_template=40):
    tpl = _templates()
    key = len(tpl)
    if _BANK.get("key") == key:
        return _BANK["F"], _BANK["y"]
    rng = np.random.default_rng(0)
    F, y = [], []
    for d, g in tpl:
        for v in [g] + _augment(g, rng, per_template):
            F.append(_features(v))
            y.append(d)
    _BANK.update(key=key, F=np.array(F, np.float32), y=np.array(y))
    return _BANK["F"], _BANK["y"]


def have_templates():
    return len(_templates()) >= 10


def class_scores(g32):
    """(probabilities over digits 0-9, best raw similarity to any template)."""
    F, y = _load_bank()
    sims = F @ _features(g32)
    scores = np.zeros(10)
    for d in range(10):
        s = np.sort(sims[y == d])[::-1][:3]
        scores[d] = s.mean() if len(s) else -1
    e = np.exp((scores - scores.max()) * 25.0)
    return e / e.sum(), float(sims.max())


def class_probs(g32):
    """Probability-like scores over digits 0-9 for one glyph (kNN on the bank)."""
    F, y = _load_bank()
    sims = F @ _features(g32)
    scores = np.zeros(10)
    for d in range(10):
        s = np.sort(sims[y == d])[::-1][:3]
        scores[d] = s.mean() if len(s) else -1
    e = np.exp((scores - scores.max()) * 25.0)
    return e / e.sum()


# --------------------------------------------------------------------------
# Number reading
# --------------------------------------------------------------------------

MIN_SCORE = 1.3           # mean best-template similarity (0-2 scale) for a line to count as "the ID number": real number lines score ~1.7-1.9, random text <= 0.9
GOOD_SCORE = 1.7
VIRTUAL_ZERO_PROB = 0.5   # a glyph the segmentation missed (a gap in the line) is probably a tiny dot = 0, but it can be any digit
#                           (at 0.98 a missed "5" became a confident "0", which flips the gender digit; in a hold-out test with jittered
#                           frames 0.5 accepted no wrong number and lost no correct one)


def _prob_matrix(img, hint=None):
    """Best reading of the number line in one photo: ((14, 10) glyph probabilities,
    glyph images, params) - or None. Every candidate line is scored by how much
    its characters look like the template digits, so the address / serial /
    dates are never mistaken for the number."""
    best = None
    for cand in segment_candidates(img, hint):
        glyphs = [glyph32(cand["gray"], b) for b in cand["boxes"]]
        rows, sims = [], []
        for g, v in zip(glyphs, cand["virtual"]):
            if v:
                row = np.full(10, (1.0 - VIRTUAL_ZERO_PROB) / 9)
                row[0] = VIRTUAL_ZERO_PROB
                rows.append(row)
            else:
                p, sim = class_scores(g)
                rows.append(p)
                sims.append(sim)
        score = float(np.mean(sims)) if sims else 0.0
        if best is None or score > best[0]:
            best = (score, np.array(rows), glyphs, cand["params"])
        if score >= GOOD_SCORE:
            break
    if best is None or best[0] < MIN_SCORE:
        return None
    return best[1], best[2], best[3]


def _valid_readings(P, validator, top=3, max_alternatives=3000, n=2):
    """The n most probable readings (digit strings) that pass `validator`,
    as [(string, log-probability)], by beam search over each glyph's top-k digits."""
    beams = [("", 0.0)]
    for p in P:
        ks = np.argsort(p)[::-1][:top]
        new = [(s + str(k), lp + float(np.log(p[k] + 1e-9))) for s, lp in beams for k in ks]
        new.sort(key=lambda t: -t[1])
        beams = new[:max_alternatives]
    return [(s, lp) for s, lp in beams if validator(s)][:n]


def read_number(imgs, validator):
    """Read the ID number from one photo, or from SEVERAL photos of the same
    card (e.g. consecutive webcam frames): every frame votes with its glyph
    probabilities (product of experts), so random blur/noise in single frames
    cancels out.
    Returns a dict {number, margin, min_prob, agreement, frames, glyphs} or None:
      margin    - how many times more probable the number is than the runner-up valid reading
      min_prob  - probability of the least certain digit of the number (0-1)
      agreement - fraction of the frames whose OWN best valid reading is this number
    validator(str) -> truthy if the 14 digits form a structurally valid ID."""
    if not have_templates():
        return None
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    logp, glyphs, frame_probs, hint, misses = None, None, [], None, 0
    for img in imgs:
        if img is None:
            continue
        res = _prob_matrix(img, hint)
        if res is None:
            misses += 1
            if not frame_probs and misses >= 2:              # the number line is not there: don't search every frame
                break
            continue
        P, g, hint = res
        frame_probs.append(P)
        lp = np.log(P + 1e-6)
        logp = lp if logp is None else logp + lp
        glyphs = glyphs or g
    if logp is None:
        return None
    logp = logp - logp.max(axis=1, keepdims=True)
    P = np.exp(logp)
    P = P / P.sum(axis=1, keepdims=True)
    best = _valid_readings(P, validator, max_alternatives=6000)
    if not best:
        return None
    number = best[0][0]
    margin = float(np.exp(best[0][1] - best[1][1])) if len(best) > 1 else float("inf")
    own = [(_valid_readings(fp, validator, max_alternatives=1500, n=1) or [("", 0)])[0][0] for fp in frame_probs]
    min_prob = float(min(P[i, int(d)] for i, d in enumerate(number)))
    return {"number": number, "margin": margin, "min_prob": min_prob, "agreement": sum(o == number for o in own) / len(own),
            "frames": len(frame_probs), "glyphs": glyphs}


def learn(glyphs, digits, max_per_class=60):
    """Save operator-confirmed glyph crops as new templates."""
    if len(glyphs) != len(digits):
        return 0
    saved = 0
    for g, d in zip(glyphs, digits):
        folder = os.path.join(TEMPLATE_DIR, str(d))
        os.makedirs(folder, exist_ok=True)
        if len(os.listdir(folder)) >= max_per_class:
            continue
        name = hashlib.md5(g.tobytes()).hexdigest()[:12] + ".png"
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            cv2.imencode('.png', g)[1].tofile(path)
            saved += 1
    return saved