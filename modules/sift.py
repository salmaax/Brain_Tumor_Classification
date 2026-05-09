import cv2
import numpy as np
from sklearn.cluster import KMeans


# initialize SIFT 
sift = cv2.SIFT_create()

VOCAB_SIZE = 300 #number of visual words in our vocabulary (this is a hyperparameter we can tune)


# Extract keypoints + descriptors For ONE image (numeric vector representation of the local region)
# keypoints :	locations of interesting points
# descriptors :	128-D vector describing each point
def get_sift_features(img):
    #ensure image is in uint8 format
    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        return [], None
    return keypoints, descriptors

# descriptors shape = (N, 128) 
# N = number of keypoints (varies per image) (so number of descriptors also varies per image)
# (this is a problem for ML models that expect fixed-size input,
# so we need to build a "vocabulary" of visual words
# to convert variable-length descriptors into fixed-length feature vectors)



# combine all descriptors from all images and group similar descriptors into clusters
# Cluster 0 => edge-like patterns
# Cluster 1 => corner-like patterns
# Cluster 2 => texture patterns
# etc
# therefore => vocabulary = "list of visual patterns in ALL images"
def build_vocabulary(descriptor_list, vocab_size= VOCAB_SIZE):
    descriptor_list = [d for d in descriptor_list if d is not None]
    #stack all descriptors into a single array
    all_descriptors = np.vstack(descriptor_list)
    
    #apply kmeans to find cluster centers (visual words)
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)
    kmeans.fit(all_descriptors)
    
    return kmeans



# many descriptors per image (variable number)
# each descriptor assigned to a “visual word” (cluster index)
# So each image becomes:
# [3, 0, 7, 1, 0, 2, ...] (each descriptor's assigned cluster index is written as a number)
# This is a histogram of counts
def image_to_feature(img, kmeans, vocab_size=VOCAB_SIZE):
    keypoints, descriptors = get_sift_features(img)
    if descriptors is None:
        return np.zeros(vocab_size)  #return empty histogram if no features found
    
    #predict which cluster each descriptor belongs or is assigned to (For THIS image, which visual word does each descriptor belong to?)
    visual_words = kmeans.predict(descriptors)
    
    #build histogram to count how many times each “visual word” appears
    feature_vector_hist = np.bincount(visual_words, minlength=vocab_size)
    
    #normalize histogram
    norm = np.linalg.norm(feature_vector_hist)
    if norm != 0:
        feature_vector_hist = feature_vector_hist / norm
    
    return feature_vector_hist




# match two images using their SIFT features
def match_images(img1, img2, top_k=30):
    kp1, des1 = get_sift_features(img1)
    kp2, des2 = get_sift_features(img2)
    
    # safety check
    if des1 is None or des2 is None:
        return None, []
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  #this creates a Brute Force matcher
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  #smaller distance = better match
    
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:top_k],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched_img, matches



def matching_accuracy(matches):
    if matches is None or len(matches) == 0:
        return 0.0

    distances = np.array([m.distance for m in matches])

    # normalize between 0 and 1
    min_d = np.min(distances)
    max_d = np.max(distances)

    if max_d == min_d:
        return 1.0

    norm_distances = (distances - min_d) / (max_d - min_d)

    similarity = 1 - np.mean(norm_distances)

    return float(similarity)


def average_matching_score(pairs):
    scores = []

    for img1, img2 in pairs:
        _, matches = match_images(img1, img2, top_k=50)
        scores.append(matching_accuracy(matches))

    return np.mean(scores)