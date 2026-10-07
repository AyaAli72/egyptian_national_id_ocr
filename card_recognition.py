"""
Egyptian National ID Card Scanner Module

This module provides functionality to scan and process Egyptian national ID cards.
It detects the card boundaries, applies perspective correction, and validates the ID number.

Main workflow:
1. Load and resize image for processing
2. Apply edge detection to find card boundaries
3. Find quadrilateral contours that could be the ID card
4. Validate front-side candidates with the template-based 14-digit reader
5. Return the best processed card image
"""

from transform import apply_perspective_correction, sort_quadrilateral_corners
import pytesseract
import numpy as np
import cv2
import imutils
import datetime
import os
import shutil
import digit_reader


def _configure_tesseract():
    """Set pytesseract to an executable path visible to this Python process."""
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
    ]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ])
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return pytesseract.pytesseract.tesseract_cmd


_TESSERACT_CMD = _configure_tesseract()


def _configure_tessdata():
    """Find a folder that contains ara.traineddata and point Tesseract at it.

    Order: TESSDATA_DIR, TESSDATA_PREFIX, <project>/tessdata, <project> itself,
    then the Tesseract install folder. TESSDATA_PREFIX is set in the environment
    (inherited by the tesseract process) instead of passing --tessdata-dir, because
    a quoted path with spaces is not passed correctly on Windows.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("TESSDATA_DIR"),
        os.environ.get("TESSDATA_PREFIX"),
        os.path.join(here, "tessdata"),
        here,
    ]
    if _TESSERACT_CMD and os.path.isfile(str(_TESSERACT_CMD)):
        candidates.append(os.path.join(os.path.dirname(str(_TESSERACT_CMD)), "tessdata"))
    for folder in candidates:
        if folder and os.path.isfile(os.path.join(folder, "ara.traineddata")):
            os.environ["TESSDATA_PREFIX"] = folder
            print(f"[Info] Arabic OCR data: {os.path.join(folder, 'ara.traineddata')}")
            return folder
    print("[Warning] ara.traineddata not found. Put it in the project's 'tessdata' folder "
          "(or set TESSDATA_DIR). Names, address and the back side cannot be read without it.")
    return None


_TESSDATA_FOLDER = _configure_tessdata()


def tessdata_dir_config():
    """Kept for compatibility: the data folder is now given through TESSDATA_PREFIX."""
    return ""


def available_langs():
    """Which of the three project models are present: ara, ara_combined, ara_number."""
    if not _TESSDATA_FOLDER:
        return []
    return [lang for lang in ("ara", "ara_combined", "ara_number")
            if os.path.isfile(os.path.join(_TESSDATA_FOLDER, lang + ".traineddata"))]


AVAILABLE_LANGS = available_langs()
if AVAILABLE_LANGS:
    print(f"[Info] Tesseract models found: {', '.join(AVAILABLE_LANGS)}")


# Configuration constants
PROCESSING_HEIGHT = 500          # Standard height for image processing
FINAL_CARD_SIZE = (1000, 630)    # Final output dimensions (width, height)
EXPECTED_ASPECT_RATIO = FINAL_CARD_SIZE[0] / FINAL_CARD_SIZE[1]
MIN_AREA_FRACTION = 0.05          # Minimum area as fraction of image (5%)
MAX_AREA_FRACTION = 0.65          # Maximum area as fraction of image (65%)
MAX_CONTOUR_CANDIDATES = 6        # Maximum number of contours to evaluate


def resize_card_to_canvas(image, target_size=FINAL_CARD_SIZE, fill_color=255):
    """
    Fit an image inside a fixed-size card canvas without changing its aspect ratio.

    Letterboxing small aspect-ratio differences keeps the downstream card
    coordinate system stable while avoiding stretching the card image.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Cannot resize an empty image")
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or color image")

    target_width, target_height = map(int, target_size)
    if target_width < 1 or target_height < 1:
        raise ValueError("Target width and height must be positive")

    source_height, source_width = image.shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * scale))))
    resized_height = max(1, min(target_height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=interpolation
    )

    canvas_shape = (target_height, target_width) + tuple(image.shape[2:])
    canvas = np.full(canvas_shape, fill_color, dtype=image.dtype)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def read_image_bgr(image_path):
    """
    Load any common image format (PNG, JPG, ...) as a 3-channel BGR array.

    PNGs may be RGBA, grayscale, palette or 16-bit. Transparent pixels are
    composited onto white so the card edges are found correctly.
    """
    # np.fromfile + imdecode also handles non-ASCII paths on Windows
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    if image.dtype == np.uint16:                      # 16-bit PNG -> 8-bit
        image = (image / 257).astype(np.uint8)

    if image.ndim == 2:                               # grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 2:                           # gray + alpha
        alpha = image[:, :, 1:2].astype(np.float32) / 255.0
        gray = (image[:, :, :1].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:                           # BGRA -> BGR on white
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        return (bgr * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def load_and_resize_image(image_path, target_height=PROCESSING_HEIGHT):
    """
    Load an image and resize it to a standard height while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the input image
        target_height (int): Target height for processing (default: 500px)
    
    Returns:
        tuple: (original_image, resized_image, scale_ratio)
            - original_image: Full resolution original image
            - resized_image: Resized image for processing
            - scale_ratio: Ratio to convert coordinates back to original size
    """
    original_image = read_image_bgr(image_path)
    
    height, width = original_image.shape[:2]
    scale_ratio = height / float(target_height)
    new_width = int(round(width / scale_ratio))
    
    resized_image = cv2.resize(
        original_image, 
        (new_width, target_height), 
        interpolation=cv2.INTER_AREA
    )
    
    return original_image.copy(), resized_image, scale_ratio

def preprocess_for_edge_detection(input_image):
    """
    Preprocess image for edge detection using adaptive Canny thresholds.
    
    Args:
        input_image: Input image (BGR format)
    
    Returns:
        tuple: (grayscale_image, edge_detected_image)
            - grayscale_image: Converted grayscale version
            - edge_detected_image: Binary edge map using Canny detection
    """
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    
    # Calculate adaptive Canny thresholds based on image statistics
    median_pixel_intensity = np.median(blurred_image)
    canny_lower_threshold = int(max(0, 0.66 * median_pixel_intensity))
    canny_upper_threshold = int(min(255, 1.33 * median_pixel_intensity))
    
    # Apply Canny edge detection
    edge_detected_image = cv2.Canny(blurred_image, canny_lower_threshold, canny_upper_threshold)
    
    return grayscale_image, edge_detected_image


def extract_and_transform_card(card_corner_points, processed_image, full_resolution_image, coordinate_scale_ratio):
    """
    Extract the ID card region and apply perspective transformation.
    
    Args:
        card_corner_points: 4-point contour representing card corners
        processed_image: Processed image used for detection
        full_resolution_image: Full resolution original image
        coordinate_scale_ratio: Ratio to scale coordinates back to original size
    
    Returns:
        tuple: (debug_visualization, perspective_corrected_card, standardized_card_image)
            - debug_visualization: Image showing detected quadrilateral
            - perspective_corrected_card: Perspective-corrected card at original resolution
            - standardized_card_image: Resized card image at standard dimensions
    """
    # Create visualization image for debugging
    grayscale_for_viz = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    debug_visualization = cv2.cvtColor(grayscale_for_viz, cv2.COLOR_GRAY2BGR)
    
    if card_corner_points is None:
        return debug_visualization, None, None

    # Draw detected quadrilateral on visualization
    cv2.drawContours(
        debug_visualization, 
        [card_corner_points.astype(int)], 
        -1, 
        (255, 0, 0),  # Blue color
        3
    )
    
    # Apply perspective transformation to original image
    full_resolution_corners = card_corner_points.reshape(4, 2) * coordinate_scale_ratio
    perspective_corrected_card = apply_perspective_correction(full_resolution_image, full_resolution_corners)
    
    # Resize to standard card dimensions
    standardized_card_image = cv2.resize(
        perspective_corrected_card, 
        FINAL_CARD_SIZE, 
        interpolation=cv2.INTER_LINEAR
    )
    
    return debug_visualization, perspective_corrected_card, standardized_card_image

def validate_id_number_region(standardized_card_image):
    """
    Read and validate the ID number from a standardized card image.

    The digit reader locates the number row and recognizes its 14 glyphs from
    the template bank, then uses the Egyptian ID structure to reject invalid
    readings. It searches the image rather than relying on a fixed crop or OCR.
    
    Args:
        standardized_card_image: Standardized card image (1000x630 pixels)
    
        Returns:
        tuple: (is_valid_id_number, extracted_digit_string)
            - is_valid_id_number: True only for a structurally valid 14-digit ID
            - extracted_digit_string: ASCII digits, or an empty string
    """
    if standardized_card_image is None or not isinstance(standardized_card_image, np.ndarray):
        return False, ""
    result = digit_reader.read_number(
        standardized_card_image,
        validator=is_valid_egyptian_id_number,
    )
    if result is not None:
        digits = result.get("number", "")
        if is_valid_egyptian_id_number(digits):
            return True, digits
    # Second reader: Tesseract's ara_number model. The template bank only holds
    # 29 glyphs from 2 cards, so on other cards / fonts it can miss the number.
    digits = read_id_with_ara_number(standardized_card_image)
    if digits:
        print("[Info] ID number read by the ara_number model")
        return True, digits
    return False, ""


_TO_ASCII_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def read_id_with_ara_number(card_image):
    """Read the ID number on a normalized 1000x630 card with the ara_number model.

    Several row positions / thresholds / sizes are read; every 14-digit reading that
    passes the ID structure check gets a vote, and the most-voted number wins.
    Returns "" if ara_number.traineddata is missing or nothing valid is read.
    """
    if "ara_number" not in AVAILABLE_LANGS or card_image is None:
        return ""
    if card_image.shape[:2] != (FINAL_CARD_SIZE[1], FINAL_CARD_SIZE[0]):
        return ""                      # only meaningful on the normalized card
    gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY) if card_image.ndim == 3 else card_image
    votes = {}
    for y0, y1 in ((470, 530), (460, 540), (480, 560), (450, 520), (490, 570)):
        crop = gray[y0:y1, 400:980]
        blurred = cv2.GaussianBlur(crop, (3, 3), 0)
        for view in (crop,
                     cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]):
            for scale in (1.0, 0.6):
                v = view if scale == 1.0 else cv2.resize(view, None, fx=scale, fy=scale,
                                                         interpolation=cv2.INTER_AREA)
                try:
                    text = pytesseract.image_to_string(
                        v, lang="ara_number",
                        config="--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩")
                except Exception as exc:
                    print(f"[Warning] ara_number OCR failed: {exc}")
                    return ""
                digits = "".join(ch for ch in text.translate(_TO_ASCII_DIGITS) if ch.isdigit())
                for i in range(0, max(1, len(digits) - 13)):
                    candidate = digits[i:i + 14]
                    if is_valid_egyptian_id_number(candidate):
                        votes[candidate] = votes.get(candidate, 0) + 1
        if votes and max(votes.values()) >= 3:
            break
    if not votes:
        return ""
    best, count = max(votes.items(), key=lambda kv: kv[1])
    return best if count >= 2 else ""      # one lucky reading is not enough


def is_valid_egyptian_id_number(id_number: str) -> bool:
    """Check the fixed-length structure, birth date, and governorate of an ID."""
    digit_translation = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789"
    )
    digits = str(id_number).translate(digit_translation)
    if len(digits) != 14 or not digits.isascii() or not digits.isdigit():
        return False
    if digits[0] not in ("2", "3"):
        return False

    year = (1900 if digits[0] == "2" else 2000) + int(digits[1:3])
    try:
        birth_date = datetime.date(year, int(digits[3:5]), int(digits[5:7]))
    except ValueError:
        return False

    valid_governorate_codes = {
        "01", "02", "03", "04",
        "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "31", "32", "33", "34", "35", "88",
    }
    return (
        birth_date <= datetime.date.today()
        and digits[7:9] in valid_governorate_codes
    )


def calculate_egyptian_id_confidence_score(id_number: str) -> float:
    """
    Calculate confidence score for Egyptian national ID number validity.
    
    Egyptian ID format: CYYMMDDSSGGG (14 digits)
    - C: Century (2=1900s, 3=2000s)
    - YY: Year of birth
    - MM: Month (01-12)
    - DD: Day (01-31)
    - SS: Sequence number
    - GGG: Governorate code + gender
    
    Scoring weights:
    - Century digit (2 or 3): 0.10
    - Valid month (01-12): 0.35
    - Valid day (01-31): 0.35
    - Valid governorate code: 0.20
    
    Args:
        id_number (str): Extracted ID number string
    
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not id_number or not id_number.isdigit():
        return 0.0

    confidence_score = 0.0

    # Validate century digit (should be 2 for 1900s or 3 for 2000s)
    if len(id_number) >= 1 and id_number[0] in ("2", "3"):
        confidence_score += 0.10

    # Validate month and day components
    if len(id_number) >= 7:
        try:
            month_str = id_number[3:5]
            day_str = id_number[5:7]
            
            if month_str.isdigit() and day_str.isdigit():
                month = int(month_str)
                day = int(day_str)
                
                # Valid month (01-12)
                if 1 <= month <= 12:
                    confidence_score += 0.35
                
                # Valid day (01-31) - basic validation
                if 1 <= day <= 31:
                    confidence_score += 0.35
        except (ValueError, IndexError):
            pass

    # Validate governorate code (positions 7-8)
    if len(id_number) >= 9:
        governorate_code = id_number[7:9]
        valid_governorate_codes = {
            "01", "02", "03", "04",  # Cairo, Alexandria, Port Said, Suez
            "11", "12", "13", "14", "15", "16", "17", "18", "19",  # Delta region
            "21", "22", "23", "24", "25", "26", "27", "28", "29",  # Upper Egypt
            "31", "32", "33", "34", "35",  # Frontier governorates
            "88"  # Born abroad
        }
        
        if governorate_code in valid_governorate_codes:
            confidence_score += 0.20

    return min(confidence_score, 1.0)

def find_best_card_contour(edge_detected_image, processed_image, full_resolution_image, coordinate_scale_ratio):
    """
    Find the best quadrilateral contour representing the ID card.
    
    This function:
    1. Finds all contours in the edge-detected image
    2. Filters for quadrilateral shapes (4 corners)
    3. Tests each candidate using the template-based national-number reader
    4. Returns the contour with the best ID number validation
    
    Args:
        edge_detected_image: Binary edge-detected image
        processed_image: Resized image used for processing
        full_resolution_image: Full resolution original image
        coordinate_scale_ratio: Scale factor for coordinate conversion
    
    Returns:
        numpy.ndarray: Best quadrilateral contour coordinates
    
    Raises:
        RuntimeError: If no suitable quadrilateral is found
    """
    # Find contours and sort by area (largest first)
    detected_contours = cv2.findContours(edge_detected_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = imutils.grab_contours(detected_contours)
    largest_contours = sorted(detected_contours, key=cv2.contourArea, reverse=True)[:5]
    
    best_validated_contour = None
    for current_contour in largest_contours:
        # Approximate contour to reduce number of points
        contour_perimeter = cv2.arcLength(current_contour, True)
        simplified_contour = cv2.approxPolyDP(current_contour, 0.02 * contour_perimeter, True)

        # Only consider quadrilaterals (4 corners)
        if len(simplified_contour) == 4:
            try:
                # Extract and transform the card region
                _, _, candidate_card_image = extract_and_transform_card(
                    simplified_contour.reshape(4, 2), 
                    processed_image, 
                    full_resolution_image, 
                    coordinate_scale_ratio
                )
                
                if candidate_card_image is not None:
                    # Validate using the 14-glyph template reader and ID structure
                    is_valid_id_number, extracted_digit_string = validate_id_number_region(candidate_card_image)
                    
                    # Calculate confidence score for additional validation
                    id_confidence_score = calculate_egyptian_id_confidence_score(extracted_digit_string)
                    
                    # Keep personal ID digits out of application logs.
                    print(
                        "[ID Validation] "
                        f"digit_count={len(extracted_digit_string)} "
                        f"valid={is_valid_id_number} "
                        f"confidence={id_confidence_score:.2f}"
                    )
                    
                    # Do not call a candidate a card based on a short digit fragment.
                    if is_valid_id_number:
                        best_validated_contour = simplified_contour
                        break
                        
            except Exception as processing_error:
                # Continue to next contour if processing fails
                print(f"[Warning] Failed to process contour: {processing_error}")
                continue
    
    if best_validated_contour is None:
        raise RuntimeError(
            "No card contour passed 14-digit ID validation. "
            "Try a clearer image and ensure digit_reader has its template bank."
        )
    
    return best_validated_contour


def is_plausible_card_quad(corners, image_shape):
    """
    Reject quadrilaterals that cannot be an ID card: thin strips, tiny regions,
    corners far outside the photo, or extreme perspective. A bad quad produces
    stretched streaks or a black image after the perspective warp.
    """
    try:
        h, w = image_shape[:2]
        quad = sort_quadrilateral_corners(np.asarray(corners, dtype="float32").reshape(4, 2))
        tl, tr, br, bl = quad
        top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
        left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
        if min(top, bottom, left, right) < 20:
            return False
        aspect = max(top, bottom) / max(left, right)
        area = cv2.contourArea(quad.reshape(-1, 1, 2))
        inside = (quad[:, 0].min() >= -0.05 * w and quad[:, 0].max() <= 1.05 * w and
                  quad[:, 1].min() >= -0.05 * h and quad[:, 1].max() <= 1.05 * h)
        return bool(
            1.25 <= aspect <= 2.0                       # card-like shape
            and area >= 0.12 * w * h                    # not a tiny/thin region
            and min(top, bottom) / max(top, bottom) >= 0.5
            and min(left, right) / max(left, right) >= 0.5
            and inside
        )
    except Exception:
        return False


def find_card_by_geometry(processed_image, coordinate_scale_ratio):
    """
    Locate the card from its SHAPE (size + aspect ratio + rectangularity).

    Unlike find_best_card_contour(), this does not need an ID number to be readable,
    so it also works for the back side of the card. Tries Canny edges and an
    Otsu foreground mask (both polarities), first strictly, then with relaxed limits.
    Returns a (4, 2) float32 array in full-resolution coordinates, or None.
    """
    h, w = processed_image.shape[:2]
    image_area = float(h * w)
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median = np.median(blur)
    edges = cv2.Canny(blur, int(max(0, 0.66 * median)), int(min(255, 1.33 * median)))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    masks = [
        edges,
        cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel),
        cv2.morphologyEx(255 - otsu, cv2.MORPH_CLOSE, kernel),
    ]

    for aspect_tol, min_rectangularity in ((0.30, 0.80), (0.45, 0.70)):   # strict, then relaxed
        best_quad, best_area = None, 0.0
        for mask in masks:
            found = imutils.grab_contours(
                cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
            for contour in sorted(found, key=cv2.contourArea, reverse=True)[:8]:
                area = cv2.contourArea(contour)
                # skip tiny blobs and the whole-frame contour
                if area < MIN_AREA_FRACTION * image_area or area > 0.97 * image_area:
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
                if min(rw, rh) < 1:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                rectangularity = area / (rw * rh)
                if abs(aspect - EXPECTED_ASPECT_RATIO) > aspect_tol or rectangularity < min_rectangularity:
                    continue

                quad = None
                perimeter = cv2.arcLength(contour, True)
                for eps in (0.02, 0.03, 0.05):
                    approx = cv2.approxPolyDP(contour, eps * perimeter, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        quad = approx.reshape(4, 2).astype("float32")
                        break
                if quad is None:
                    quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")

                if not is_plausible_card_quad(quad, processed_image.shape):
                    continue
                # The card is light; a wallet, phone or table is dark. Without this the
                # biggest dark rectangle wins and temp_back.jpg comes out black.
                inside = np.zeros(gray.shape, np.uint8)
                cv2.fillConvexPoly(inside, quad.astype(np.int32), 255)
                brightness = cv2.mean(gray, mask=inside)[0]
                if brightness < 100:
                    continue
                score = min(area / image_area, 0.6) + brightness / 255.0
                if score > best_area:
                    best_quad, best_area = quad, score
        if best_quad is not None:
            return best_quad * coordinate_scale_ratio
    return None


def correct_upside_down(card_image):
    """
    Return the card the right way up.

    Photos of the back are often taken upside down, and Tesseract cannot read upside-down
    Arabic at all: religion and marital status then come back as "غير محدد" and the job as
    fragments. The card is read both ways up and the orientation with more confident Arabic
    text is kept. If the OCR check fails for any reason, the card is returned unchanged.
    """
    def readable_text(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        binary = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        config = " ".join(
            part for part in (tessdata_dir_config(), "--psm 6") if part
        )
        data = pytesseract.image_to_data(binary, lang="ara", config=config,
                                         output_type=pytesseract.Output.DICT)
        return sum(float(conf) for conf, text in zip(data["conf"], data["text"])
                   if float(conf) >= 40 and len(str(text).strip()) >= 2)

    try:
        rotated = cv2.rotate(card_image, cv2.ROTATE_180)
        normal_score, rotated_score = readable_text(card_image), readable_text(rotated)
        if rotated_score > 1.5 * normal_score and rotated_score - normal_score > 150:
            print(f"[Info] card was upside down - rotated 180 degrees "
                  f"(text score {normal_score:.0f} -> {rotated_score:.0f})")
            return rotated
    except Exception as orientation_error:
        print(f"[Warning] orientation check skipped: {orientation_error}")
    return card_image


def _card_detection_failure_message(card_side_type, full_frame_digit_count=None, full_frame_valid=False):
    if card_side_type == "front":
        if full_frame_digit_count is None:
            id_detail = "The template reader could not read an ID number from the full image."
        elif full_frame_digit_count != 14:
            id_detail = (
                f"The full-image reader returned {full_frame_digit_count} digits; "
                "an Egyptian ID must contain exactly 14."
            )
        elif full_frame_valid:
            id_detail = "The ID number was read, but the card outline could not be found."
        else:
            id_detail = (
                "The full-image reader returned 14 digits, but they failed ID structure "
                "validation."
            )
        return (
            "No card outline was detected. "
            f"{id_detail} Make sure the card fills most of the image, the ID "
            "number is sharp and unobstructed, and either all four card edges "
            "are visible or the image is cropped tightly to the card."
        )

    return (
        "No card outline was detected. Make sure the entire card is "
        "visible with all four edges in frame, or crop the image to the card."
    )


def scan_id_card(input_image_path, card_side_type, _try_rotated=True):
    """
    Main function to scan and process an Egyptian national ID card.
    
    This function performs the complete scanning workflow:
    1. Load and resize the input image
    2. Apply edge detection to find card boundaries
    3. Detect the best quadrilateral contour representing the card
    4. Apply perspective transformation to get a top-down view
    5. Save the processed card image
    
    Args:
        input_image_path (str): Path to the input image file
        card_side_type (str): Either "front" or "back" to specify card side
    
    Returns:
        numpy.ndarray: Processed card image (1000x630 pixels)
    
    Raises:
        ValueError: If image cannot be loaded
        RuntimeError: If no suitable card contour is found
    """
    if card_side_type not in ("front", "back"):
        raise ValueError(
            f"Invalid card_side_type: {card_side_type}. Must be 'front' or 'back'"
        )

    if card_side_type == "front" and _try_rotated:
        try:
            return scan_id_card(input_image_path, card_side_type, _try_rotated=False)
        except RuntimeError as first_error:
            # The number row can only be read upright: try the photo turned 180 degrees.
            rotated_path = os.path.splitext(input_image_path)[0] + "_rot180.png"
            cv2.imencode(".png", cv2.rotate(read_image_bgr(input_image_path), cv2.ROTATE_180))[1].tofile(rotated_path)
            try:
                print("[Info] front not found upright - trying it rotated 180 degrees")
                return scan_id_card(rotated_path, card_side_type, _try_rotated=False)
            except RuntimeError:
                raise first_error
            finally:
                os.remove(rotated_path)

    # Step 1: Load and resize image for processing
    full_resolution_image, processing_sized_image, coordinate_scale_ratio = load_and_resize_image(
        input_image_path, 
        PROCESSING_HEIGHT
    )
    
    # Step 2: look for the card outline in the photo first.
    detected_corners = None
    final_processed_card = None
    try:
        if card_side_type == "back":
            # The back has no ID number; never select it using the front-side number reader.
            corners = find_card_by_geometry(processing_sized_image, coordinate_scale_ratio)
        else:
            try:
                _, edge_detected_image = preprocess_for_edge_detection(processing_sized_image)
                corners = find_best_card_contour(
                    edge_detected_image, processing_sized_image,
                    full_resolution_image, coordinate_scale_ratio
                ).reshape(4, 2) * coordinate_scale_ratio
            except RuntimeError:
                corners = find_card_by_geometry(processing_sized_image, coordinate_scale_ratio)
    except Exception as e:
        print(f"[Warning] Card outline search failed: {e}")
        corners = None

    if corners is not None and is_plausible_card_quad(corners, full_resolution_image.shape):
        h, w = full_resolution_image.shape[:2]
        quad_fraction = cv2.contourArea(np.asarray(corners, np.float32).reshape(-1, 1, 2)) / float(h * w)
        if quad_fraction < 0.85:          # a real card inside a bigger photo
            detected_corners = corners
            final_processed_card = apply_perspective_correction(
                full_resolution_image, corners, output_size=FINAL_CARD_SIZE
            )

    # Step 3: no outline (or it is the whole frame) -> the image is already a cropped/scanned card.
    full_card = resize_card_to_canvas(full_resolution_image)
    if final_processed_card is None:
        h, w = full_resolution_image.shape[:2]
        looks_like_card = 1.40 <= w / float(h) <= 1.75
        ok, digits = False, ""
        if card_side_type == "front":
            ok, digits = validate_id_number_region(full_card)
            print(f"[Full-image check] digit_count={len(digits)} valid={ok}")
        if (card_side_type == "front" and ok) or (card_side_type == "back" and looks_like_card):
            final_processed_card = full_card
        else:
            raise RuntimeError(f"{card_side_type} side: " + _card_detection_failure_message(
                card_side_type, len(digits) if card_side_type == "front" else None, ok))

    # Debug picture: red outline = detected card (none = whole photo used)
    try:
        debug_view = processing_sized_image.copy()
        if detected_corners is not None:
            pts = (np.asarray(detected_corners) / coordinate_scale_ratio).astype(np.int32)
            cv2.polylines(debug_view, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
        cv2.imwrite(f"debug_{card_side_type}_detect.jpg", debug_view)
    except Exception:
        pass

    # Safety net: a (nearly) black result means a wrong region was cropped
    if final_processed_card.mean() < 25:
        print("[Warning] Cropped card is almost black -> using the whole image instead")
        final_processed_card = full_card

    # Step 4: make sure the card is the right way up before it is saved and read
    # (the front is already upright: its number row was read, which only works upright)
    if card_side_type == "back":
        final_processed_card = correct_upside_down(final_processed_card)

    # Save processed image based on card side
    if card_side_type == "front":
        output_file_path = "temp_front.jpg"
    elif card_side_type == "back":
        output_file_path = "temp_back.jpg"
    
    cv2.imwrite(output_file_path, final_processed_card)
    print(f"[Success] Processed {card_side_type} side saved as {output_file_path}")
    
    return final_processed_card