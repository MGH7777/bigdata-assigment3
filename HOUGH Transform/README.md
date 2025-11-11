

This repo contains two small command-line tools for detecting the heart center in the **herzinfarkt** images and saving visual panels of the results.

- **`hough_heart_detector.py`** — Circle-only detector (part **a** of the assignment). Uses a Hough-based circle search plus simple scoring to pick the best circle and mark the center. Handles error images by writing a panel that says **NO CIRCLE FOUND**.
- **`hough_heart_detector2.py`** — Circle **or** ellipse detector (part **b**). Tries a circle first; if needed, also tests ellipse candidates and chooses the better one. Panels say **Best shape overlay**.

Both tools accept **DICOM** (`.dcm` or files with DICM magic) and **raster images** (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).

---

## ¨

1) Install

Use Python 3.10+ and install the requirements:

```bash
pip install -r requirements.txt

2) Run

Run the app, you will be asked to provide the path to your images

Then you will be asked to provide the output folder