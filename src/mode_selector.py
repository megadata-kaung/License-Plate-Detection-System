# this one on ice as well

import cv2

def is_vehicle_image(img):
    h, w, _ = img.shape

    # Heuristic: vehicle images are large
    if w > 800 or h > 600:
        return True

    return False
