import cv2
from plate_detection import crop_plate
from vehicle_mode import detect_plate_vehicle
from ocr_reader import read_plate_text
from state_classifier import classify_state
from mode_selector import is_vehicle_image
from gui import *


image_path = "../data/sample/military.jpg"

img = cv2.imread(image_path)
if img is None:
    print("Image not found")
    exit()

# Decide mode
if is_vehicle_image(img):
    plate_img = detect_plate_vehicle(img)
else:
    plate_img = crop_plate(image_path)

if plate_img is None:
    print("Plate not detected")
    exit()

plate_text = read_plate_text(plate_img)
state = classify_state(plate_text)

print("Detected Plate:", plate_text)
print("State:", state)

cv2.imshow("Detected Plate", plate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

