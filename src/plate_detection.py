import cv2
import numpy as np
from morphology import apply_morphology

# Contours are extracted from the morphologically processed image.
# Candidate regions are filtered based on geometric properties such as area and aspect ratio
# to identify the license plate.


def detect_plate(image_path):
    """
    Kaung and Haider worked on the detection logic.
    We're looking for a rectangular shape that matches the 
    typical Aspect Ratio of a license plate (wide and short).
    """

    binary, _, closed = apply_morphology(image_path)

    # findContours helps us find all the "blobs" in the cleaned image.
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    plate_contour = None
    max_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)

        # Plate-like constraints
        if area > 5000 and 2.0 < aspect_ratio < 6.0:
            if area > max_area:
                max_area = area
                plate_contour = cnt

    return plate_contour, contours, closed


def crop_plate(image_path):
    """
    Crop detected license plate region from the original image
    """

    plate_contour, _, _ = detect_plate(image_path)
    image = cv2.imread(image_path)

    if plate_contour is None:
        return None

    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = image[y:y+h, x:x+w]

    return plate_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "../data/sample/test-7.jpg"
    plate_contour, contours, closed = detect_plate(image_path)
    original = cv2.imread(image_path)

    if plate_contour is not None:
        x, y, w, h = cv2.boundingRect(plate_contour)
        cv2.rectangle(original, (x,y), (x+w,y+h), (0,0,255), 2)

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(closed, cmap="gray")
    plt.title("Morphology Output")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Detected Plate")
    plt.axis("off")

    plt.show()

# different Main method for testing

# if __name__ == "__main__":
#     image_path = "../data/sample/test-4.jpg"
#
#     plate_contour, contours, closed = detect_plate(image_path)
#     original = cv2.imread(image_path)
#
#     # Draw all contours
#     debug_all = original.copy()
#     cv2.drawContours(debug_all, contours, -1, (0, 255, 0), 2)
#
#     # Draw detected plate
#     if plate_contour is not None:
#         x, y, w, h = cv2.boundingRect(plate_contour)
#         cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 3)
#
#     plate_img = crop_plate(image_path)
#
#     cv2.imshow("All Contours", debug_all)
#     cv2.imshow("Detected Plate", original)
#
#     if plate_img is not None:
#         cv2.imshow("Cropped Plate", plate_img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
