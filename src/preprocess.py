import cv2
import numpy as np

# loading the image
def preprocess_image(image_path):
    """
    Initial cleaning phase. 
    We convert to Grayscale and use a Bilateral Filter.
    Saif chose Bilateral because it removes noise but keeps 
    the edges of the letters sharp (unlike Gaussian blur which smudges them).
    """
    # removed color information and kept only intensity which is dark vs light
    # license plates are defined by contrast not colors
    # text is dark and plate is bright

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)

    return image, gray, denoised




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img, gray, denoised = preprocess_image("../data/sample/test-7.jpg")

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(denoised, cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

    plt.show()

# different testing method

# if __name__ == "__main__":
#     img, gray, denoised = preprocess_image("../data/sample/test-2.jpg")
#
#     cv2.imshow("Original", img)
#     cv2.imshow("Grayscale", gray)
#     cv2.imshow("Denoised", denoised)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
