import cv2
import pytesseract
import re

from matplotlib import pyplot as plt
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def clean_text(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text

def ocr_pass(image, config):
    text = pytesseract.image_to_string(image, config=config)
    return clean_text(text)

def read_plate_text(plate_img):
    """
    Multi-pass OCR:
    Try light → medium → aggressive
    Return the best non-empty result
    """

    if plate_img is None:
        return ""

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # PASS 1: Minimal processing (keeps previous success)
    pass1 = ocr_pass(
        gray,
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    if len(pass1) >= 3:
        return pass1

    # PASS 2: Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    pass2 = ocr_pass(
        thresh,
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    if len(pass2) >= 3:
        return pass2

    # PASS 3: Inverted (black plates)
    inverted = cv2.bitwise_not(thresh)

    pass3 = ocr_pass(
        inverted,
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    return pass3

def visualize_ocr_pipeline(plate_img):
    import matplotlib.pyplot as plt
    from pytesseract import Output

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    thresh_adapt = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )

    _, thresh_otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    images = [
        ("Cropped Plate", plate_img),
        ("Grayscale", gray),
        ("Adaptive Threshold", thresh_adapt),
        ("Otsu Threshold", thresh_otsu)
    ]

    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (title, img) in zip(axs, images):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.show()

    # OCR bounding boxes (use adaptive threshold)
    data = pytesseract.image_to_data(
        thresh_adapt,
        output_type=Output.DICT,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(thresh_adapt, cmap="gray")
    ax.set_title("OCR Detection + Confidence")
    ax.axis("off")

    for i in range(len(data["text"])):
        if int(data["conf"][i]) > 30 and data["text"][i].strip():
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i]
            )
            rect = plt.Rectangle((x, y), w, h,
                                 fill=False, edgecolor="red", linewidth=1)
            ax.add_patch(rect)
            ax.text(
                x, y - 3,
                f"{data['text'][i]} ({data['conf'][i]})",
                color="yellow",
                fontsize=8,
                backgroundcolor="black"
            )

    plt.show()


