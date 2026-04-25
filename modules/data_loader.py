import cv2
import os

# load data using folder traversal
def load_data(base_path, img_size=(256,256)):
    images = []
    labels = []
    
    classes = os.listdir(base_path)
    
    for label in classes:
        class_path = os.path.join(base_path, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue # skip if image is not loaded properly / corrupted
            images.append(img)
            labels.append(label)
            
    return images, labels