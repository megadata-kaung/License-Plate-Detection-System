import cv2
import numpy as np
from plate_detection import crop_plate


def segment_characters(image_path):
    """
    Haider worked on this part to chop the plate into individual letters.
    We used contour detection and some geometry rules to make sure we only 
    grab the actual characters and not the plate screws or borders.
    """
    plate_img = crop_plate(image_path)
    if plate_img is None:
        return [], None, None

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 1. Strong adaptive threshold (keep strokes)
    # Saif suggested Mean-C here because it helps keep the character strokes thick
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        5
    )

    # 2. CLOSE to reconnect broken characters (CRITICAL)
    # Kaung noticed some letters were breaking apart, so we used Morphology (Closing)
    # to glue those tiny gaps back together.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Remove very thick horizontal structures (plate border)
    # This is a cool trick we found to wipe out the plate frame so it doesn't mess up our contours.
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    border = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    binary = cv2.subtract(binary, border)

    # 4. Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    plate_h, plate_w = binary.shape
    char_regions = []

    # This loop is where we filter out the noise. 
    # We're checking for height and width ratios so we don't pick up random spots.
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = h / float(w)

        # Logic: Characters should be tall, not too wide, and take up most of the plate height.
        if (
                0.4 * plate_h < h < 0.95 * plate_h and
                w < 0.30 * plate_w and
                (
                        w > 0.025 * plate_w or
                        aspect_ratio > 2.5  # allow thin tall characters like '1' or '7'
                )
        ):
            char_regions.append((x, y, w, h))

    # Sort left → right so the plate reads correctly and not in random order!
    char_regions = sorted(char_regions, key=lambda b: b[0])

    characters = []
    for x, y, w, h in char_regions:
        char = binary[y:y+h, x:x+w]
        # Adding a little padding (border) so the OCR has some "breathing room"
        char = cv2.copyMakeBorder(
            char, 5, 5, 5, 5,
            cv2.BORDER_CONSTANT, value=0
        )
        # Resizing everything to 40x60 so the classifier gets a consistent shape
        char = cv2.resize(char, (40, 60))
        characters.append(char)

    return characters, binary, plate_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    characters, binary, plate_img = segment_characters("../data/sample/test-7.jpg")

    plt.figure(figsize=(6,3))
    plt.imshow(binary, cmap="gray")
    plt.title("Binary Plate")
    plt.axis("off")
    plt.show()

    for i, char in enumerate(characters):
        plt.figure()
        plt.imshow(char, cmap="gray")
        plt.title(f"Character {i+1}")
        plt.axis("off")
        plt.show()

