"""OCR extraction adapter for the geometric front-field and digit readers.

Front-side field locations come from detect_fields (anchored to the detected
national-number row); the number itself comes from digit_reader, with the
ara_number Tesseract model as a second reader.

The three Tesseract models in the project's tessdata folder are all used:
  ara           - Arabic text (names, address, back side)
  ara_combined  - second Arabic model; both are read and the more confident
                  reading wins (its Persian letter forms are converted to Arabic)
  ara_number    - digits: the ID number fallback and the house number in the address

The module keeps the globals/functions expected by main.py so the scanner can
be integrated without changing the API response contract.
"""
import re
from difflib import SequenceMatcher

import cv2
import numpy as np
import pytesseract

import arabic_names
import card_recognition
import detect_fields
import gender
import pob


extracted_id_number = ""
extracted_full_name = ""
extracted_first_name = ""
extracted_father_name = ""
extracted_address = ""
extracted_birth_date = ""
extracted_gender = ""
extracted_governorate = ""
extracted_religion = ""
extracted_marital_status = ""
extracted_spouse_name = ""
extracted_job_title = ""
extracted_workplace = ""
extraction_errors = []

_FRONT_TEXT_FIELDS = ("first_name", "father_name", "address")
_BACK_REGIONS = {
    # Coordinates are relative to the scanner's 1000 x 630 normalized card.
    "job": (230, 70, 820, 140),
    "workplace": (230, 125, 820, 190),
    # gender, religion and marital status are printed on ONE line: read it as one
    # crop and pick the words out, instead of two overlapping fixed boxes.
    "status_line": (150, 175, 900, 265),
    "spouse": (150, 255, 900, 320),
}

# Every value the status line can contain -> the spelling we return.
_STATUS_WORDS = {
    "gender": {"ذكر": "ذكر", "انثي": "أنثى"},
    "religion": {"مسلم": "مسلم", "مسلمه": "مسلمة", "مسيحي": "مسيحي", "مسيحيه": "مسيحية"},
    "marital": {"اعزب": "أعزب", "انسه": "آنسة", "متزوج": "متزوج", "متزوجه": "متزوجة",
                "مطلق": "مطلق", "مطلقه": "مطلقة", "ارمل": "أرمل", "ارمله": "أرملة"},
}


def reset_extracted_data():
    """Clear all values between uploads."""
    global extracted_id_number, extracted_full_name, extracted_first_name
    global extracted_father_name, extracted_address, extracted_birth_date
    global extracted_gender, extracted_governorate, extracted_religion
    global extracted_marital_status, extracted_spouse_name, extracted_job_title
    global extracted_workplace, extraction_errors

    extracted_id_number = ""
    extracted_full_name = ""
    extracted_first_name = ""
    extracted_father_name = ""
    extracted_address = ""
    extracted_birth_date = ""
    extracted_gender = ""
    extracted_governorate = ""
    extracted_religion = ""
    extracted_marital_status = ""
    extracted_spouse_name = ""
    extracted_job_title = ""
    extracted_workplace = ""
    extraction_errors = []


def _read_image(path):
    try:
        return card_recognition.read_image_bgr(path)
    except Exception as exc:
        extraction_errors.append(f"Could not load normalized card image: {exc}")
        return None


def _clean_text(text):
    text = str(text or "").replace("\x0c", " ")
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F0-9A-Za-z\s،,.؛:/-]", " ", text)
    return " ".join(text.split())


def _norm(text):
    text = re.sub(r"[\u064B-\u0652\u0640]", "", text)
    return text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))


# ara_combined writes Persian forms of some letters/digits; the card is Arabic.
_SCRIPT_FIX = str.maketrans({
    "ی": "ي", "ې": "ي", "ۍ": "ي", "ک": "ك", "ہ": "ه", "ە": "ه", "ھ": "ه", "پ": "ب",
    "۰": "٠", "۱": "١", "۲": "٢", "۳": "٣", "۴": "٤", "۵": "٥", "۶": "٦", "۷": "٧", "۸": "٨", "۹": "٩",
    "\u200f": "", "\u200e": "", "_": " ", "“": "", "”": "", '"': "", "'": "",
})
_TO_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _text_langs():
    """Arabic text models to read with (both when present)."""
    langs = [l for l in ("ara", "ara_combined") if l in card_recognition.AVAILABLE_LANGS]
    return langs or ["ara"]


def _fix_script(text):
    return _clean_text(str(text or "").translate(_SCRIPT_FIX))


def _keep_token(conf, token):
    """Confident Arabic words; short tokens (a digit, 'ش') only when very confident."""
    if re.search(r"[\u0621-\u064A]{2,}", token):
        return conf >= 30
    return conf >= 60 and bool(re.search(r"[\u0621-\u064A\u0660-\u06690-9]", token))


def _ocr_views(crop, scales=(1.0, 0.6)):
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale only when the TEXT would be too small. Every back box is < 100 px high,
    # so the old "height < 100" rule doubled 30-px text to 60 px and Tesseract lost it.
    if gray.shape[0] < 45:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    views = []
    # Tesseract reads best at ~20-25 px letter height; card text is ~30-35 px on the
    # 1000x630 card, so also try a reduced copy (crisp scans read NOTHING at full size).
    for scale in scales:
        g = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        g = cv2.copyMakeBorder(g, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
        blurred = cv2.GaussianBlur(g, (3, 3), 0)
        views += [
            g,
            cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12),
        ]
    return views


def _tess_data(image, lang, psm):
    try:
        return pytesseract.image_to_data(image, lang=lang, config=f"--oem 1 --psm {psm}",
                                         output_type=pytesseract.Output.DICT)
    except Exception as exc:
        if not any("Tesseract OCR unavailable" in item for item in extraction_errors):
            extraction_errors.append(f"Arabic Tesseract OCR unavailable ({lang}): {exc}")
        return None


def _ocr_readings(crop, psms=(6,), keep_short=False):
    """Every (score, text) reading over models x views x psms.
    score = summed confidence of the kept words (ara and ara_combined compete)."""
    readings = []
    for scale in (1.0, 0.6):
        # the reduced copy is only needed when the full-size reading is not clearly good
        if readings and max(r[2] for r in readings) >= 85:
            break
        readings += _readings_at(crop, scale, psms, keep_short)
    return [(score, text) for score, text, _ in readings]


def _readings_at(crop, scale, psms, keep_short):
    readings = []
    for view in _ocr_views(crop, scales=(scale,)):
        for lang in _text_langs():
            for psm in psms:
                data = _tess_data(view, lang, psm)
                if data is None:
                    return readings
                words = []
                for conf, token in zip(data["conf"], data["text"]):
                    token = _fix_script(token)
                    if not token:
                        continue
                    letters = len(re.findall(r"[\u0621-\u064A]", token))
                    keep = _keep_token(float(conf), token) if keep_short else (
                        # short words (<= 3 letters) must be confident: drops stray "كل", "الا"
                        letters >= 2 and float(conf) >= (30 if letters > 3 else 70))
                    if keep:
                        words.append((float(conf), token))
                if words:
                    readings.append((sum(c for c, _ in words), " ".join(t for _, t in words),
                                     sum(c for c, _ in words) / len(words)))
    return readings


def _best_text(crop, psms=(6,), keep_short=False):
    readings = _ocr_readings(crop, psms, keep_short)
    return max(readings)[1] if readings else ""


def _back_ocr(crop, psm=6):
    """Back side: the reading with the most CONFIDENT Arabic words (not the longest text)."""
    return _best_text(crop, (psm,))


def _house_number(line_gray):
    """Read the house/building number at the start (right end) of an address line with
    ara_number. The Arabic models misread it (٢٠٧ -> ٠١ / ٧); ara_number reads digits.
    Returns ASCII digits or ""."""
    if "ara_number" not in card_recognition.AVAILABLE_LANGS:
        return ""
    otsu = cv2.threshold(cv2.GaussianBlur(line_gray, (3, 3), 0), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    digit_boxes, letter_rights = [], []
    for lang in _text_langs():
        data = _tess_data(otsu, lang, 7)
        if data is None:
            return ""
        for i, token in enumerate(data["text"]):
            token = _fix_script(token)
            if not token or float(data["conf"][i]) < 0:
                continue
            left, right = data["left"][i], data["left"][i] + data["width"][i]
            if re.search(r"[\u0660-\u06690-9]", token):
                digit_boxes.append((left, right))
            elif re.fullmatch(r"[\u0621-\u064A\s]+", token):
                letter_rights.append(right)
    if not digit_boxes:
        return ""                      # this address has no number
    h = line_gray.shape[0]
    digit_left = min(l for l, _ in digit_boxes)
    before = [r for r in letter_rights if r <= digit_left + 5]
    start = max(before + [digit_left - h]) + 2
    number_crop = line_gray[:, max(0, start):]
    if number_crop.shape[1] < 8:
        return ""
    votes = {}
    for scale in (1.0, 2.0):
        v = number_crop if scale == 1.0 else cv2.resize(number_crop, None, fx=scale, fy=scale,
                                                        interpolation=cv2.INTER_CUBIC)
        try:
            text = pytesseract.image_to_string(
                v, lang="ara_number",
                config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩")
        except Exception:
            return ""
        digits = "".join(ch for ch in text.translate(card_recognition._TO_ASCII_DIGITS) if ch.isdigit())
        if 1 <= len(digits) <= 5:
            votes[digits] = votes.get(digits, 0) + 1
    return max(votes.items(), key=lambda kv: kv[1])[0] if votes else ""


def _line_crops(fields, field_name, gray):
    """One crop per text line of a front field (lines found by detect_fields)."""
    box = fields.get("fields", {}).get(field_name)
    if box is None:
        return []
    fx, fy, fw, fh = box
    H, W = gray.shape[:2]
    crops = []
    for (x, y, w, h) in fields.get("lines", []):
        if fy <= y + h / 2 <= fy + fh and fx <= x + w / 2 <= fx + fw:
            px, py = int(0.8 * h), int(0.35 * h)
            crops.append(gray[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)])
    return crops


def _read_front_field(fields, field_name, gray):
    """Read a front field line by line (a whole multi-line block often drops a line)."""
    lines = _line_crops(fields, field_name, gray)
    if not lines:                                  # fall back to the whole field box
        crop = _field_crop(fields, field_name)
        return _best_text(crop, (6,), keep_short=(field_name == "address"))
    texts = []
    for index, line in enumerate(lines):
        text = _best_text(line, (7,), keep_short=(field_name == "address"))
        if field_name == "address" and index == 0:
            number = _house_number(line)
            if number:
                words = [w for w in text.split() if not re.search(r"[\u0660-\u06690-9]", w)]
                text = " ".join([number.translate(_TO_ARABIC_DIGITS)] + words)
        if text:
            texts.append(text)
    return " ".join(texts)


def _match_status(words):
    """Pick gender / religion / marital status out of the words read on the status line."""
    found = {}
    for word in words:
        key = _norm(word)
        # each word belongs to the ONE field it matches best ("انثي" is close to "انسه" too)
        ratio, field, value = max(
            (SequenceMatcher(None, key, v).ratio(), f, canon)
            for f, vocab in _STATUS_WORDS.items() for v, canon in vocab.items()
        )
        if ratio >= 0.75 and ratio > found.get(field, (0, ""))[0]:
            found[field] = (ratio, value)
    return {field: value for field, (_, value) in found.items()}


def _field_crop(result, field_name):
    box = result.get("fields", {}).get(field_name)
    color = result.get("color")
    if box is None or color is None:
        return None
    x, y, w, h = map(int, box)
    H, W = color.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return color[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else None


def _set_derived_id_fields(number):
    global extracted_birth_date, extracted_gender, extracted_governorate
    if len(number) != 14 or not number.isascii() or not number.isdigit():
        return
    year = (1900 if number[0] == "2" else 2000) + int(number[1:3])
    extracted_birth_date = f"{year:04d}/{number[3:5]}/{number[5:7]}"
    extracted_gender = gender.gen(number)
    extracted_governorate = pob.placeOfBirth(number)


def extract_front_side_data(image_path):
    """Read the front ID number, then OCR the geometrically detected text bands."""
    global extracted_id_number, extracted_full_name, extracted_first_name
    global extracted_father_name, extracted_address

    image = _read_image(image_path)
    if image is None:
        return

    valid, extracted_id_number = card_recognition.validate_id_number_region(image)
    if valid:
        _set_derived_id_fields(extracted_id_number)
    else:
        extracted_id_number = ""
        extraction_errors.append("The front-side ID number could not be read as a valid 14-digit number.")

    fields = detect_fields.detect(image)
    if fields.get("problem"):
        extraction_errors.append(f"Front-side text-field detection: {fields['problem']}")
        return

    frame_gray = cv2.cvtColor(fields["color"], cv2.COLOR_BGR2GRAY)
    found = {}
    for name in _FRONT_TEXT_FIELDS:
        text = _read_front_field(fields, name, frame_gray)
        if name in ("first_name", "father_name") and text:
            text = arabic_names.fix_name_spelling(text)
        found[name] = text

    extracted_first_name = found.get("first_name", "")
    extracted_father_name = found.get("father_name", "")
    extracted_address = found.get("address", "")
    extracted_full_name = " ".join(
        value for value in (extracted_first_name, extracted_father_name) if value
    )
    for name, value in found.items():
        if not value:
            extraction_errors.append(f"Front-side {name.replace('_', ' ')} text was not recognized.")


def _back_text_lines(gray):
    """Text lines anywhere on the back card: (x, y, w, h), top to bottom.
    Found from dark ink rows, so it does not depend on fixed box positions."""
    H, W = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    paper = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    margin = int(0.02 * W)
    best = []
    for diff, dark in ((60, 120), (45, 150), (35, 255)):          # strict -> loose ink test
        ink = (((paper.astype(np.int16) - g) > diff) & (g < dark)).astype(np.uint8)
        ink[:, :margin] = 0
        ink[:, W - margin:] = 0
        rows = np.convolve(ink.sum(axis=1).astype(np.float32), np.ones(3) / 3, mode="same")
        on = rows > max(3.0, 0.01 * W)
        bands, start = [], None
        for y, v in enumerate(list(on) + [False]):
            if v and start is None:
                start = y
            elif not v and start is not None:
                bands.append([start, y])
                start = None
        merged = []
        for band in bands:                         # dots above/below letters split a line
            if merged and band[0] - merged[-1][1] <= 6:
                merged[-1][1] = band[1]
            else:
                merged.append(band)
        lines = []
        for y0, y1 in merged:
            if not 14 <= y1 - y0 <= 90:            # too thin = noise, too tall = barcode/photo
                continue
            cols = np.where(ink[y0:y1].sum(axis=0) > 0)[0]
            if len(cols) == 0 or cols[-1] - cols[0] < 40:
                continue
            lines.append((int(cols[0]), y0, int(cols[-1] - cols[0] + 1), y1 - y0))
        if len(lines) > len(best):
            best = lines
        if len(lines) >= 3:
            break
    return best


def _line_crop(gray, box):
    x, y, w, h = box
    H, W = gray.shape[:2]
    px, py = int(0.6 * h), int(0.35 * h)
    return gray[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]


def _read_back_by_lines(gray):
    """Layout-free back reading: find the status line by its WORDS, then take the job and
    workplace from the lines above it and the spouse from the line below.
    Returns a dict, or None when no status line was found."""
    lines = _back_text_lines(gray)
    if not lines:
        return None
    best_index, best_status = None, {}
    for index, box in enumerate(lines):
        words = [w for _, text in _ocr_readings(_line_crop(gray, box), (7,)) for w in text.split()]
        status = _match_status(words)
        if len(status) > len(best_status):
            best_index, best_status = index, status
        if len(status) == 3:
            break
    if best_index is None:
        return None

    status_box = lines[best_index]
    sx, sy, sw, line_h = status_box
    # Up to 2 text lines right above the status line = job, workplace. Lines that are not
    # Arabic words (the ID number, a date, pattern noise) are skipped.
    texts, lowest_y = [], sy
    for box in reversed(lines[:best_index]):       # nearest first
        if lowest_y - (box[1] + box[3]) > 2.5 * line_h or len(texts) == 2:
            break
        text = _best_text(_line_crop(gray, box), (7,))
        if len(re.findall(r"[\u0621-\u064A]", text)) >= 3:
            texts.append(text)
        lowest_y = box[1]
    texts.reverse()                                # top to bottom
    result = dict(best_status)
    result["job"] = texts[0] if texts else ""
    result["workplace"] = texts[1] if len(texts) > 1 else ""
    result["spouse"] = ""
    if result.get("marital", "").startswith("متزوج"):
        pitch = 1.6 * line_h
        if best_index > 0:
            pitch = max(1.2 * line_h, min(2.5 * line_h, sy - lines[best_index - 1][1]))
        below = None
        if best_index + 1 < len(lines) and lines[best_index + 1][1] - (sy + line_h) <= 2.5 * line_h:
            below = _line_crop(gray, lines[best_index + 1])
        spouse = _best_text(below, (7,)) if below is not None else ""
        if not spouse:
            # the spouse line can touch the barcode and not be found as a line:
            # read the place one line-step below the status line, right-aligned like it
            H, W = gray.shape[:2]
            y0, y1 = int(sy + pitch - 0.35 * line_h), int(sy + pitch + 1.35 * line_h)
            x0, x1 = int(sx + sw - max(2.0 * sw, 400)), int(sx + sw + 0.6 * line_h)
            spouse = _best_text(gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)], (7,))
        result["spouse"] = spouse
    return result


def extract_back_side_data(image_path):
    """OCR the back side. First layout-free (find the status line by its words); if that
    fails, fall back to the fixed regions of the normalized card."""
    global extracted_job_title, extracted_workplace, extracted_religion
    global extracted_marital_status, extracted_spouse_name

    image = _read_image(image_path)
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    def crop(name):
        x0, y0, x1, y1 = _BACK_REGIONS[name]
        return gray[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]

    found = _read_back_by_lines(gray)
    if found and (found.get("religion") or found.get("marital")):
        status = found
        extracted_job_title = found.get("job", "")
        extracted_workplace = found.get("workplace", "")
        spouse = found.get("spouse", "")
    else:
        # fallback: fixed positions
        extracted_job_title = _back_ocr(crop("job"))
        extracted_workplace = _back_ocr(crop("workplace"))
        words = [w for _, text in _ocr_readings(crop("status_line"), (7, 6)) for w in text.split()]
        status = _match_status(words)
        spouse = _back_ocr(crop("spouse")) if status.get("marital", "").startswith("متزوج") else ""

    extracted_religion = status.get("religion", "")
    extracted_marital_status = status.get("marital", "")
    extracted_spouse_name = re.sub(r"^\s*ال?زوج[هة]?\S*\s*[:：]?\s*", "", spouse or "").strip()
    if status.get("gender") and extracted_gender and status["gender"] != extracted_gender:
        extraction_errors.append(
            f"Back-side gender '{status['gender']}' does not match the ID number ('{extracted_gender}').")

    # (no workplace is normal - housewives, students without one - so it is not an error)
    for name, value in (("job", extracted_job_title), ("religion", extracted_religion),
                        ("marital status", extracted_marital_status)):
        if not value:
            extraction_errors.append(f"Back-side {name} text was not recognized.")
