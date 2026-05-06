# License Plate Recognition (LPR) and State Identification System

This project presents the development of a License Plate Recognition (LPR) and State Identification System. The system is designed to automatically detect Malaysian vehicle license plates from images, recognize the alphanumeric characters using OCR, and identify the registered Malaysian state based on predefined license plate prefixes.

The project was developed using classical image processing and computer vision techniques without using machine learning, deep learning, Haar Cascade classifiers, or template matching methods. :contentReference[oaicite:0]{index=0}

---

# Features

- Automatic vehicle license plate detection
- Character segmentation from detected plates
- OCR-based alphanumeric recognition
- Malaysian state identification system
- Graphical User Interface (GUI)
- Image preprocessing and morphology pipeline
- Support for multiple vehicle types
- Visualization of intermediate processing stages

---

# Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Tesseract OCR

---

# Project Structure

```text
project/
│
├── data/
│   ├── sample/
│   └── templates/
│
├── src/
│   ├── app.py
│   ├── char_segmentation.py
│   ├── generate_templates.py
│   ├── gui.py
│   ├── main.py
│   ├── mode_selector.py
│   ├── morphology.py
│   ├── ocr_reader.py
│   ├── plate_detection.py
│   ├── preprocess.py
│   ├── segmentation.py
│   ├── state_classifier.py
│   └── vehicle_mode.py
│
├── requirements.txt
├── README.md
└── .gitignore


System Pipeline

The developed system follows the following processing pipeline:

Image preprocessing
Grayscale conversion
Thresholding using Otsu’s algorithm
Morphological operations
License plate detection
Character segmentation
OCR recognition using Tesseract
Malaysian state classification
Image Processing Techniques Used
Grayscale Conversion

Converts RGB images into grayscale to reduce computational complexity while preserving important structural information.

Otsu’s Thresholding

Automatically binarizes grayscale images to separate foreground and background regions.

Morphological Operations

Used to improve image quality and enhance character visibility.

Morphological Opening

Removes small noise regions and unwanted artifacts.

Morphological Closing

Reconnects broken character strokes and fills gaps.

Contour Detection

Detects potential license plate regions using contour analysis.

Geometric Filtering

Filters contours based on aspect ratio and area constraints to identify likely plate regions.

Character Segmentation

Separates individual characters from the detected plate using connected component analysis.

OCR Recognition

Uses Tesseract OCR to recognize alphanumeric characters from segmented plates.

State Classification

Identifies Malaysian states using predefined license plate prefix rules.

Supported Vehicle Types

The system was tested using multiple vehicle categories including:

Private vehicles
Government vehicles
Military vehicles
Taxis
Jeeps
Trucks
Buses
Experimental Results

The system performs effectively when:

the license plate is clearly visible
lighting conditions are moderate
the plate region is properly detected

The system may struggle under:

extreme lighting conditions
blurred images
distant vehicles
tilted plates
complex backgrounds

Sample Results
Original Image

<img width="911" height="553" alt="image" src="https://github.com/user-attachments/assets/083b72a1-845e-4d25-9f5c-0a50b5183f7c" />


Preprocessing Output

<img width="938" height="448" alt="image" src="https://github.com/user-attachments/assets/327e095d-d760-4bea-bb4e-a1bf1cea0a21" />


Morphology Output

<img width="941" height="444" alt="image" src="https://github.com/user-attachments/assets/0a1fd330-542e-4f28-bba1-3d9e1fdd6fbe" />


OCR Result and State Classification Result

<img width="938" height="439" alt="image" src="https://github.com/user-attachments/assets/8b01980e-2d26-4f21-b758-76d0f2bef547" />


Limitations
Sensitive to lighting conditions
OCR accuracy depends heavily on plate localization
Difficulty detecting very small or distant plates
Performance decreases with blurred or tilted plates
Limited dataset size
Future Improvements

Potential future improvements include:

Integration of YOLO or SSD for plate detection
Deep learning based OCR systems
CNN-based character recognition
Adaptive preprocessing techniques
Larger and more diverse datasets
Confidence-based OCR validation
How to Run
Clone Repository
git clone <your-repository-link>
Install Dependencies
pip install -r requirements.txt
Run the Project
python src/main.py
GUI

The system includes a graphical user interface (GUI) for:

image upload
preprocessing visualization
plate detection visualization
OCR result display
state classification result display
Contributors
Mohammad Haider Iftikhar
Saif Ahmad
Htet Kaung Myat Oo
Academic Information

Module:
Image Processing, Computer Vision and Pattern Recognition (CT036-3-IPPR)

Institution:
Asia Pacific University of Technology and Innovation (APU)

References

This project references concepts and techniques from:

Digital Image Processing
Computer Vision
OCR systems
Morphological image processing
Contour detection algorithms
Malaysian license plate classification systems
