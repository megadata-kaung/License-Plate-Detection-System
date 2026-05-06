# putting this script on ice

import cv2
import numpy as np

def detect_plate_vehicle(img):
    h_img, w_img, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 120, 240)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_plate = None
    best_score = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        aspect = w / float(h)
        area = w * h

        # Geometry constraints (TIGHTER)
        if not (4.0 < aspect < 5.8):
            continue

        if area < 4000 or area > 40000:
            continue

        # Position bias (lower half of image)
        if y < h_img * 0.4:
            continue

        plate_candidate = img[y:y+h, x:x+w]

        # Color contrast check
        roi_gray = cv2.cvtColor(plate_candidate, cv2.COLOR_BGR2GRAY)
        contrast = roi_gray.std()

        if contrast > best_score:
            best_score = contrast
            best_plate = plate_candidate

    return best_plate

