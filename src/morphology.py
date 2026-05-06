import cv2
import numpy as np
from segmentation import segment_image

# Small decorative elements such as country labels were intentionally removed during morphological
# processing to avoid interfering with plate detection.
# State identification is performed later using rule-based logic after isolating the plate region.


def apply_morphology(image_path):
    """
    Saif set this up to clean up the binary image.
    We used Opening to kill the small white noise (salt) 
    and Closing to fill the holes in the plate/letters.
    """

    _, binary = segment_image(image_path)

    # We used a 5x5 rectangle kernel - standard stuff from the lectures.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Opening: remove small noise (erosion then dilation)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Closing: fill gaps and connect regions (dilation then erosion)
    # This makes the plate look like one solid block for easier detection.
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return binary, opened, closed

# shows matplot results this one
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    binary, opened, closed = apply_morphology("../data/sample/car.jpg")

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(binary, cmap="gray")
    plt.title("Binary")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(opened, cmap="gray")
    plt.title("After Opening")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(closed, cmap="gray")
    plt.title("After Closing")
    plt.axis("off")

    plt.show()

# this is a different method for testing

# if __name__ == "__main__":
#     binary, opened, closed = apply_morphology("../data/sample/test-2.jpg")
#
#     cv2.imshow("Binary", binary)
#     cv2.imshow("After Opening", opened)
#     cv2.imshow("After Closing", closed)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
