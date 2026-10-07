# Egyptian ID card scanner — updated recognition flow

## What changed

- Front-side card candidates are accepted only when `digit_reader.py` finds and
  validates a structurally valid 14-digit Egyptian national ID. The old fixed
  pixel ID crop and digit Tesseract OCR have been removed.
- `detect_fields.py` locates the front-side name and address bands relative to
  the detected number row. Tesseract transcribes those detected crops; it no
  longer depends on fixed front-side name/address boxes.
- `id_card_data_extractor.py` restores the extractor interface expected by
  `main.py`, derives birth date, gender, and governorate from the validated ID,
  and keeps the existing API response field names.
- Back-side text still uses Arabic Tesseract OCR on the established normalized
  card regions.

## Run

Install Python dependencies with `pip install -r requirements.txt`, install
Tesseract separately, and make the Arabic (`ara`) trained data available to it.
Then start the API from this directory:

```sh
uvicorn main:app --reload --port 8000
```

On Windows, the scanner checks the standard Tesseract installation paths.
For a custom installation, set `TESSERACT_CMD` to the full path of
`tesseract.exe` before starting the API.

### Arabic data without administrator access (Windows PowerShell)

The scanner accepts `TESSDATA_DIR` as the direct path to a folder containing
`ara.traineddata`. For example:

```powershell
$data = Join-Path $env:LOCALAPPDATA "Tesseract\tessdata"
New-Item -ItemType Directory -Force -Path $data | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ara.traineddata" `
  -OutFile (Join-Path $data "ara.traineddata")
$env:TESSDATA_DIR = $data
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --tessdata-dir $data --list-langs
```

Confirm the list includes `ara`, then launch Uvicorn from this same PowerShell
window so it inherits `TESSDATA_DIR`.

The ID-number reader uses the built-in digit template bank in `digit_reader.py`;
it does not need Tesseract's numeric language data. Arabic OCR is still needed
to transcribe names, addresses, and back-side text.

## Validation status

The updated files pass Python compilation, module/API imports, template-bank
availability, ID-structure checks, the new validator adapter contract, and
blank-image detection checks. No sample card images were included, so real-card
recognition accuracy has not been exercised here.

## v5 bug fixes

- Photos are no longer used whole: the card outline is searched first, and the
  whole image is used only when it is already a cropped card (no outline found,
  or the outline covers 85%+ of the frame).
- The back is no longer cropped to a dark object (wallet, phone, table): the
  card must be light inside (mean brightness 100 or more).
- An upside-down front is retried rotated 180 degrees.
- The back is not flipped unless the rotated reading is clearly better.
- Gender, religion and marital status are read from one status-line crop and
  matched to the known values, so they come back as e.g. "مسلمة" / "آنسة".
- Back crops are no longer blown up 2x, and a reduced copy is also read.
- The spouse is read only for married people. The back gender is checked
  against the ID number.
- `python main.py` (web mode) now starts. The CLI result includes `errors`.

## Tesseract models (tessdata folder)

The three models go in the project's `tessdata` folder (included in this zip).
The code finds them by itself, so no TESSDATA_PREFIX setup or admin rights are needed.

| Model | Used for |
|---|---|
| `ara.traineddata` | names, address, back side (required) |
| `ara_combined.traineddata` | second Arabic reader: both are read, the more confident reading wins |
| `ara_number.traineddata` | backup ID-number reader (when the digit templates fail) and the house number in the address |

Only `ara` is required. Without the other two the scanner still works, but it loses
the backup ID reader and reads house numbers less well.
