import cv2
import numpy as np
from preprocess import preprocess_image


def segment_image(image_path):
    # Kaung bro we need to apply segmentation here adaptive thresholding 
    # you can find it on YouTube as well as the lecture vids how to apply
    # bro i turned the image into pure black and white
    # white text and pixels are the one we care about which is the text and plate edges
    # turn the background to black - looks like a photocopy lol

    # because in the image lighting sometimes is uneven, a global threshold might fail.
    # Adaptive threshold adjusts locally so it works even in shadows.
    """
    Perform adaptive thresholding using Gaussian weights to handle lighting issues.
    """

    _, _, denoised = preprocess_image(image_path)

    # Saif used Gaussian Thresholding here because it's smoother than simple mean.
    # We used THRESH_BINARY_INV so the text becomes white (easier for contours).
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5
    )

    return denoised, binary


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    denoised, binary = segment_image("../data/sample/taxi.jpg")

    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(denoised, cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(binary, cmap="gray")
    plt.title("Adaptive Threshold")
    plt.axis("off")

    plt.show()

# this code is another main method for different testing

# if __name__ == "__main__":
#     denoised, binary = segment_image("../data/sample/test-2.jpg")
#
#     cv2.imshow("Denoised", denoised)
#     cv2.imshow("Binary (Segmented)", binary)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
