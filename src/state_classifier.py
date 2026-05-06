import cv2
import os
import numpy as np

TEMPLATE_DIR = "../data/templates"
CHAR_SIZE = (40, 60)


def load_templates():
    templates = {}

    for file in os.listdir(TEMPLATE_DIR):
        if file.endswith(".png"):
            label = os.path.splitext(file)[0]
            path = os.path.join(TEMPLATE_DIR, file)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, CHAR_SIZE)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            img = cv2.bitwise_not(img)

            templates[label] = img

    return templates


def recognize_character(char_img, templates, is_prefix=False):
    """
    Recognize a single character using template matching.
    If is_prefix=True, apply domain-aware prefix correction.
    """

    char_img = cv2.resize(char_img, CHAR_SIZE)
    _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)
    char_img = cv2.bitwise_not(char_img)

    scores = []

    for label, template in templates.items():
        result = cv2.matchTemplate(
            char_img, template, cv2.TM_SQDIFF_NORMED
        )
        score = result[0][0]
        scores.append((label, score))

    # Sort by best (lowest) score
    scores = sorted(scores, key=lambda x: x[1])

    if len(scores) < 2:
        return None

    best_char, best_score = scores[0]
    second_char, second_score = scores[1]

    # 🚨 Confidence gate
    if best_score > 0.6:
        return None

    # ---------------- DEBUG: PROVE V IS CONSIDERED ----------------
    print("Top 5 matches:")
    for label, sc in scores[:5]:
        print(f"  {label} -> {round(sc, 3)}")
    # --------------------------------------------------------------

    # Shape features
    h, w = char_img.shape
    aspect = h / float(w)
    white_ratio = np.sum(char_img == 255) / char_img.size

    # --- General disambiguation rules ---

    # Z vs 2
    if best_char == "Z" and second_char == "2":
        if aspect > 1.4:
            best_char = "2"

    # U vs S
    if best_char == "U" and second_char == "S":
        if white_ratio < 0.38:
            best_char = "S"

    # B vs V (shape-based)
    if best_char == "B" and second_char == "V":
        if aspect > 1.6:
            best_char = "V"

    # --- PREFIX-ONLY DOMAIN FIX ---
    # If this is the prefix position, allow V override if close enough
    if is_prefix and best_char in {"B", "8"}:
        for label, sc in scores[:5]:
            if label == "V" and abs(sc - best_score) < 0.08:
                best_char = "V"
                break

    print("Chosen match:", best_char, "score:", round(best_score, 3))
    return best_char


def recognize_plate(char_images):
    templates = load_templates()
    plate_text = ""

    for idx, char_img in enumerate(char_images):
        char = recognize_character(
            char_img,
            templates,
            is_prefix=(idx == 0)
        )

        if char is None:
            continue

        # First 1–2 positions → letters
        if idx < 2 and char.isdigit():
            continue

        # Remaining positions → digits
        if idx >= 2 and char.isalpha():
            continue

        plate_text += char

    return plate_text


def classify_state(plate_text):
    """
        Haider wrote this classifier logic.
        It's basically a dictionary look-up or pattern matcher.
        We check the prefix or keywords in the OCR text to see
        which state the plate belongs to.
        """

    if not plate_text or len(plate_text) == 0:
        return "Unknown"

    prefix = plate_text[0]  # first character

    state_map = {
        "V": "Selangor (Malaysia)",
        "W": "Kuala Lumpur (Malaysia)",
        "J": "Johor (Malaysia)",
        "A": "Perak (Malaysia)",
        "P": "Penang (Malaysia)",
        "M": "Malacca (Malaysia)",
        "N": "Negeri Sembilan (Malaysia)",
        "C": "Pahang (Malaysia)",
        "D": "Kelantan (Malaysia)",
        "T": "Terengganu (Malaysia)",
        "K": "Kedah (Malaysia)",
        "R": "Perlis (Malaysia)",
        "S": "Sabah (Malaysia)",
        "Q": "Sarawak (Malaysia)",
        "Z": "Military / Special Series"
    }

    return state_map.get(prefix, "Malaysia (Other State)")

