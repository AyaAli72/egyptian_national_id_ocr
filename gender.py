# gender.py

def gen(id_str):
    # Guard for empty/short strings
    if not id_str or len(id_str) < 13:
        return "غير معروف"

    # 13th digit (0-based index 12)
    gender_ch = id_str[12]

    # Minimal handling: try ASCII int first
    try:
        d = int(gender_ch)
    except ValueError:
        # Handle single Arabic-Indic digit; otherwise give up gracefully
        arabic_indic_map = {
            '٠': 0, '١': 1, '٢': 2, '٣': 3, '٤': 4,
            '٥': 5, '٦': 6, '٧': 7, '٨': 8, '٩': 9
        }
        if gender_ch in arabic_indic_map:
            d = arabic_indic_map[gender_ch]
        else:
            return "غير معروف"

    return "أنثى" if (d % 2 == 0) else "ذكر"
