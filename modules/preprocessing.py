import cv2



def resize_image(img, img_size=(256, 256)):
    return cv2.resize(img, img_size)

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# resize data and convert to grayscale to reduce noise and computational load
def standardize(img, img_size=(256, 256)):
    img = to_grayscale(img)
    img = resize_image(img, img_size)
    return img


# gaussian filter (smoothing, reduces noise and small variations but can blur edges)
# Gaussian blur replaces each pixel with a weighted average of its neighbors
def gaussian_filter(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# median filter (replaces each pixel with the median of its neighbors, preserves edges better than Gaussian)
def median_filter(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def filter_image(img, method="gaussian", ksize=5):
    if method == "gaussian":
        return gaussian_filter(img, ksize)
    elif method == "median":
        return median_filter(img, ksize)
    else:
        raise ValueError("Invalid method. Choose 'gaussian' or 'median'")
    
    
def preprocess(img, img_size=(256, 256), method="gaussian", ksize=5):
    img = standardize(img, img_size)
    img = filter_image(img, method, ksize)
    return img