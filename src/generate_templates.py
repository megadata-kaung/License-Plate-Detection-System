import cv2
import numpy as np
import os
import string

# Output directory (templates are in data/)
TEMPLATE_DIR = "../data/templates"
CHAR_SIZE = (40, 60)
os.makedirs(TEMPLATE_DIR, exist_ok=True)
# Make sure directory exists
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Characters to generate
characters = list(string.ascii_uppercase) + list("0123456789")

for ch in characters:
    # Create blank image
    img = np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)

    # Draw character (thick font)
    cv2.putText(
        img,
        ch,
        (3, 50),                     # position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,                          # scale
        255,                          # white
        3,                            # thickness (IMPORTANT)
        cv2.LINE_AA
    )

    # Slight blur to mimic real plate texture
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Threshold back to binary
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Save
    filename = f"{ch}.png"
    cv2.imwrite(os.path.join(TEMPLATE_DIR, filename), img)

print("Templates regenerated successfully.")
