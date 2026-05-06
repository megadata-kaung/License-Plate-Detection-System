import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from plate_detection import crop_plate
from vehicle_mode import detect_plate_vehicle
from mode_selector import is_vehicle_image
from ocr_reader import read_plate_text, visualize_ocr_pipeline
from state_classifier import classify_state
# from ocr_reader import visualize_ocr


class LPRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IPPR – Malaysian License Plate Recognition")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.image_path = None
        self.original_img = None
        self.plate_img = None

        self.build_ui()

    # ================= UI =================
    def build_ui(self):
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        # ===== SIDEBAR =====
        sidebar = tk.Frame(main_frame, width=200, bg="#2b2b2b")
        sidebar.pack(side="left", fill="y")

        tk.Label(
            sidebar,
            text="LPR Controls",
            fg="white",
            bg="#2b2b2b",
            font=("Segoe UI", 14, "bold"),
        ).pack(pady=20)

        tk.Button(
            sidebar, text="📂 Load Image", command=self.load_image, width=18
        ).pack(pady=10)

        tk.Button(
            sidebar, text="▶ Run Detection", command=self.run_detection, width=18
        ).pack(pady=10)

        tk.Button(
            sidebar,
            text="📊 Show Processing",
            command=self.show_processing_steps,
            width=18,
        ).pack(pady=10)

        # ===== CONTENT =====
        content = tk.Frame(main_frame, bg="#1e1e1e")
        content.pack(side="right", fill="both", expand=True)

        tk.Label(
            content,
            text="Malaysian License Plate Recognition System",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=15)

        panel_frame = tk.Frame(content, bg="#1e1e1e")
        panel_frame.pack()

        self.input_panel = self.create_image_panel(
            panel_frame, "Input Image", 500
        )
        self.input_panel.pack(side="left", padx=20)

        self.plate_panel = self.create_image_panel(
            panel_frame, "Detected Plate", 350
        )
        self.plate_panel.pack(side="left", padx=20)

        self.result_label = tk.Label(
            content,
            text="Plate: —\nState: —",
            fg="white",
            bg="#1e1e1e",
            font=("Segoe UI", 14),
        )
        self.result_label.pack(pady=20)

        # ===== STATUS BAR =====
        self.status = tk.Label(
            self.root,
            text="Ready",
            anchor="w",
            bg="#111111",
            fg="white",
            padx=10,
        )
        self.status.pack(side="bottom", fill="x")

    def create_image_panel(self, parent, title, width):
        frame = tk.Frame(parent, bg="#2b2b2b", bd=2, relief="groove")

        tk.Label(
            frame,
            text=title,
            bg="#2b2b2b",
            fg="white",
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=5)

        label = tk.Label(frame, bg="black")
        label.pack(padx=10, pady=10)

        frame.label = label
        frame.display_width = width

        return frame

    # ================= LOGIC =================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return

        self.image_path = path
        self.original_img = cv2.imread(path)
        self.plate_img = None

        self.show_image(self.original_img, self.input_panel)
        self.plate_panel.label.config(image="")
        self.result_label.config(text="Plate: —\nState: —")

        self.status.config(text="Image loaded")

    def run_detection(self):
        if self.original_img is None:
            return

        if is_vehicle_image(self.original_img):
            self.status.config(text="Vehicle image detected")
            self.plate_img = detect_plate_vehicle(self.original_img)
        else:
            self.status.config(text="Close-up plate detected")
            self.plate_img = crop_plate(self.image_path)

        if self.plate_img is None:
            self.result_label.config(text="Plate: Not detected\nState: Unknown")
            self.status.config(text="Plate not detected")
            return

        self.show_image(self.plate_img, self.plate_panel)

        plate_text = read_plate_text(self.plate_img)
        state = classify_state(plate_text)

        self.result_label.config(
            text=f"Plate: {plate_text if plate_text else '—'}\nState: {state}"
        )
        self.status.config(text="OCR completed")

    # ================= MATPLOTLIB =================
    def show_processing_steps(self):
        if self.original_img is None:
            return

        # ================== PREPROCESSING ==================
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Grayscale")
        plt.imshow(gray, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Binary (Otsu Threshold)")
        plt.imshow(binary, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # ================== MORPHOLOGY ==================
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("After Opening")
        plt.imshow(opening, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("After Closing")
        plt.imshow(closing, cmap="gray")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # ================== PLATE DETECTION ==================
        if self.plate_img is not None:
            plt.figure(figsize=(6, 3))
            plt.title("Detected Plate Region")
            plt.imshow(cv2.cvtColor(self.plate_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        # ================== CHARACTER SEGMENTATION ==================
        try:
            from char_segmentation import segment_characters
            chars, _, _ = segment_characters(self.image_path)
        except Exception:
            chars = []

        if chars:
            cols = min(6, len(chars))
            rows = (len(chars) + cols - 1) // cols

            plt.figure(figsize=(12, 3))
            for i, ch in enumerate(chars[:12]):
                plt.subplot(1, min(12, len(chars)), i + 1)
                plt.imshow(ch, cmap="gray")
                plt.title(f"Char {i + 1}")
                plt.axis("off")

            plt.suptitle("Character Segmentation Output", fontsize=14)
            plt.tight_layout()
            plt.show()

            if self.plate_img is not None:
                visualize_ocr_pipeline(self.plate_img)


    # ================= DISPLAY =================
    def show_image(self, img, panel):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        scale = panel.display_width / w
        resized = cv2.resize(img_rgb, (panel.display_width, int(h * scale)))

        im = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(im)

        panel.label.imgtk = imgtk
        panel.label.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = LPRApp(root)
    root.mainloop()


