# Multi-Class Brain Tumor Classification using SIFT, BoVW & Boosting

---


Brain tumor classification from MRI images is a critical task in medical diagnostics. Manual analysis is time-consuming and subject to human error.

This project builds an automated system that classifies MRI images into multiple tumor categories using classical computer vision and machine learning techniques.

Goals:
- Extract meaningful visual patterns from MRI scans
- Build an interpretable, structured pipeline
- Evaluate traditional CV methods (SIFT + BoVW)

---

## Architecture

MRI Image  
→ Preprocessing (Gaussian + Median Filtering)  
→ Segmentation (K-Means Clustering)  
→ Feature Extraction (SIFT → Bag of Visual Words)  
→ Feature Vector (Histogram Representation)  
→ Classification (Gradient Boosting)  
→ Predicted Tumor Class  

---

## How It Works

### 1. Preprocessing
- Gaussian filtering reduces global noise
- Median filtering removes impulse noise while preserving edges

### 2. Segmentation
- K-Means clustering groups pixel intensities
- Highlights relevant regions in MRI images

### 3. Feature Extraction
- SIFT detects keypoints and extracts local descriptors
- K-Means builds a visual vocabulary
- Each image is converted into a histogram of visual words (BoVW)

### 4. Classification
- Gradient Boosting classifier trained on BoVW features
- Captures non-linear relationships between features

---

## Results

- Accuracy: ~78%
- Strong performance on No Tumor class
- Some confusion between tumor subtypes (expected)

Insight:  
The model effectively separates normal vs tumor cases, while fine-grained tumor classification remains challenging due to visual similarity.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/classification.ipynb


## Tech Stack

- Python
- OpenCV
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
