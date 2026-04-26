import cv2
import numpy as np


# automatically finds optimal threshold and splits image into:
# foreground (bright regions) , background (dark regions)
def threshold_segmentation(img):
    #safety check
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #otsu's thresholding
    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    return binary




# groups pixels based on intensity into k regions (k distinct intensity levels)
def kmeans_segmentation(img, k=3):
    #reshape image to 2d
    reshaped_img = img.reshape((-1, 1))
    reshaped_img = np.float32(reshaped_img)
    #define stopping criteria 
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0
        )
    # apply kmeans
    # labels = tells which cluster each pixel belongs to
    # centers = tells what value (intensity color) that cluster represents
    _, labels, centers = cv2.kmeans(
        reshaped_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
    #reshape back to original
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img.shape)
    return segmented