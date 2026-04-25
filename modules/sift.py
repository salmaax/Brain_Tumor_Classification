import cv2

# initialize SIFT 
sift = cv2.SIFT_create()



# Extract keypoints + descriptors (numeric vector representation of the local region)
def get_sift_features(img):
    #ensure image is in uint8 format
    if img.dtype != "uint8":
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors




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