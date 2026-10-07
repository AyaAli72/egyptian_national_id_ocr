# Egyptian National ID Scanner (v5)

Reads the front and back of an Egyptian national ID card from photos or scans and
returns the card data as JSON. Runs as a FastAPI web service or from the command line.

| Side | Fields |
|---|---|
| Front | national ID number, full name, address |
| From the ID number | birth date, gender, place of birth (governorate) |
| Back | job, workplace, religion, marital status, spouse name |

---

## Project structure

```
egyptian_id_scanner/
├── main.py                    FastAPI web service + command-line entry point
├── card_recognition.py        finds the card in the photo, crops/straightens it, reads the ID number
├── digit_reader.py            template-based reader for the 14-digit ID number
├── detect_fields.py           finds the name / address lines on the front (anchored to the ID row)
├── id_card_data_extractor.py  OCR of the front text and the back side
├── arabic_names.py            fixes common OCR spelling slips in Arabic names
├── transform.py               perspective correction
├── gender.py                  gender from the 13th digit of the ID number
├── pob.py                     governorate from digits 8-9 of the ID number
├── debug_back.py              shows what the back-side reader sees (for troubleshooting)
├── requirements.txt
└── tessdata/
    ├── ara.traineddata
    ├── ara_combined.traineddata
    └── ara_number.traineddata
```

---

## Requirements

- **Python 3.10 or newer**
- **Tesseract OCR 5** installed on the computer
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
    (default path `C:\Program Files\Tesseract-OCR\tesseract.exe` is found automatically;
    for another path set `TESSERACT_CMD` to the full path of `tesseract.exe`)
  - Linux: `sudo apt install tesseract-ocr`
- The three Arabic models in the project's `tessdata` folder (included)

### Tesseract models

The code finds the `tessdata` folder by itself. No `TESSDATA_PREFIX` setup and no
administrator rights are needed.

| Model | Used for | Required |
|---|---|---|
| `ara.traineddata` | names, address, back-side text | **yes** |
| `ara_combined.traineddata` | second Arabic reader: every line is read with both models and the more confident reading wins (its Persian letter forms ی ک ۷ are converted to Arabic ي ك ٧) | no |
| `ara_number.traineddata` | backup ID-number reader, and the house number in the address | no |

Without the two optional models the scanner still works, but it has no backup ID reader
and reads house numbers less reliably.

Search order for the models: `TESSDATA_DIR` → `TESSDATA_PREFIX` → `<project>/tessdata` →
`<project>` → the Tesseract install folder.

---

## Installation

### Windows (PowerShell)

```powershell
cd C:\path\to\egyptian_id_scanner
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Linux / macOS

```bash
cd egyptian_id_scanner
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On start-up you should see:

```
[Info] Arabic OCR data: ...\tessdata\ara.traineddata
[Info] Tesseract models found: ara, ara_combined, ara_number
```

---

## Usage

### Command line (one card)

```powershell
python main.py --front front.jpg --back back.jpg
```

The result is printed and saved as `result_<date>_<time>.json`. The cropped cards are
saved as `temp_front.jpg` and `temp_back.jpg`.

### Web service

```powershell
python main.py
```

or

```powershell
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs**, choose **POST /api/upload → Try it out**, pick the
front and back images and press **Execute**.

| Endpoint | Description |
|---|---|
| `POST /api/upload` | upload `front` and `back` images (PNG/JPG), returns the extracted data |
| `GET /api/result/{session_id}` | a saved result |
| `GET /api/result/{session_id}/download` | the saved result as a JSON file |
| `GET /api/temp/{front\|back}` | the last processed (cropped) card image |
| `GET /health` | health check |

A web page can be added as `static/index.html` (served at `/`).

### Example result

```json
{
  "fields": {
    "id_number": "30207300200646",
    "name_ar": "آية علي كمال علي إبراهيم",
    "address": "٢٠٧ ش الاسكندرانى محرم بك الاسكندرية",
    "birth_date": "2002/07/30",
    "gender": "أنثى",
    "religion": "مسلمة",
    "social_status": "آنسة",
    "husband_name": "لا يوجد",
    "job": "طالبة",
    "place_of_work": "جامعة الإسكندرية",
    "place_of_birth": "الإسكندرية"
  },
  "errors": []
}
```

`errors` lists every field that could not be read, so an empty value is never silent.

---

## How it works

### 1. Finding the card (`card_recognition.py`)

1. The card outline is searched first: edges and light/dark masks, then the shape must be
   card-like (aspect ratio 1.25–2.0, big enough, inside the photo).
2. **Front:** the outline whose ID number reads as a valid 14-digit number wins.
   **Back:** the outline must be lighter than its surroundings, so a dark wallet, phone or
   table is never mistaken for the card.
3. If no outline is found (or it covers 85%+ of the image), the image is treated as an
   already-cropped card. If even that fails, the front reports an error; the back is read
   from the whole image with a warning (the back never stops the scan).
4. The card is straightened to 1000 × 630 px.
5. **Orientation:** an upside-down front is retried rotated 180°; an upside-down back is
   detected by comparing how much confident Arabic text each orientation gives.

### 2. ID number

1. `digit_reader.py` finds the number row and recognises each of the 14 glyphs against a
   built-in template bank.
2. If that fails, the `ara_number` model reads the row at several positions and sizes;
   a number is accepted only when at least 2 readings agree.
3. Every candidate must pass the ID structure check: century digit 2/3, a real birth date
   not in the future, and a valid governorate code.

Birth date, gender (13th digit: odd = ذكر, even = أنثى) and governorate (digits 8–9) are
derived from the validated number.

### 3. Front text (`detect_fields.py` + `id_card_data_extractor.py`)

- Name and address lines are located relative to the ID-number row (no fixed boxes).
- Each line is read separately with `ara` and `ara_combined`, on the plain, Otsu and
  adaptive-threshold images; the reading with the highest total word confidence wins.
- The house number at the start of the address is read with `ara_number`.
- Known first names get their hamza / ة / ى spelling fixed (`arabic_names.py`).

### 4. Back text (layout-free)

1. Every text line on the back is found from dark ink rows, wherever it is printed.
2. The **status line** is the line containing gender / religion / marital words
   (e.g. `أنثى مسلمة آنسة`); words are fuzzy-matched to the known values:
   - gender: ذكر، أنثى
   - religion: مسلم، مسلمة، مسيحي، مسيحية
   - marital status: أعزب، آنسة، متزوج، متزوجة، مطلق، مطلقة، أرمل، أرملة
3. The 1–2 Arabic lines right above it are the **job** and **workplace**
   (number/date lines are skipped).
4. The line below it is the **spouse**, read only when the status is married.
5. If no status line is found, fixed regions of the normalized card are used as a fallback.
6. If the gender on the back differs from the gender in the ID number, it is reported in
   `errors`.

---

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ara.traineddata not found` | models missing or misnamed | put the 3 files in `tessdata/`; check Windows did not add `.txt` |
| `Error opening data file ... ara.traineddata` | old code version | use v5; it sets the data folder itself |
| `front side: No card outline was detected` | ID number not readable | sharper photo, whole card visible, number row not covered or reflecting |
| `back side: no card outline found - reading the whole image` | back edges not visible | still works; for best results show all four edges or crop the card |
| `No module named 'cv2'` | virtual environment not active | `.\venv\Scripts\Activate.ps1` |
| back fields empty or wrong | unusual layout / photo | run `python debug_back.py` (see below) |

### debug_back.py

```powershell
python debug_back.py                  # uses temp_back.jpg from the last run
python debug_back.py path\to\back.jpg
```

Prints every text line found on the back, what Tesseract reads on it and which status words
matched, and saves `debug_back_lines.jpg` with the lines drawn on the card.

---

## Known limitations

- Back-side reading has been tested on generated back cards, not yet on many real ones.
- The built-in digit template bank is small (29 glyphs from 2 cards); `ara_number` is the
  backup when it fails.
- A spouse line printed over the barcode may be read only partly.
- One scan (front + back) takes about 10–15 seconds, because each line is read with two models.
- **Privacy:** `temp_front.jpg` / `temp_back.jpg` are shared by all requests, `/api/temp/...`
  returns the last processed card without a session check, and uploaded images in
  `results/` are never deleted. Fix these before running the service for other people or
  on a network.

---

## Changelog

### v5
- Photos are no longer used whole: the card outline is searched first.
- The back is no longer cropped to a dark object; the card brightness is judged against
  its surroundings, so dim photos also work.
- The back never stops the scan; without an outline the whole image is read.
- Upside-down front is retried rotated 180°; the back is flipped only when clearly better.
- Back side read layout-free (status line found by its words); results are matched to the
  known values instead of returning raw OCR text.
- All three Tesseract models used: `ara` + `ara_combined` compete, `ara_number` is the backup
  ID reader and reads the house number.
- The `tessdata` folder is found automatically (no `TESSDATA_PREFIX`, no admin rights).
- Front name/address read line by line; back crops no longer blown up 2×.
- Spouse read only for married people; back gender checked against the ID number.
- `python main.py` (web mode) starts correctly; command-line results include `errors`.
- Added `debug_back.py`.

### v4
- Template-based ID-number reader (`digit_reader.py`) and field detection anchored to
  the ID row (`detect_fields.py`).
