import numpy as np
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm

def HS_color_profile(image):
    
    image = cv2.imread(image)
    
    # convert it to HS
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # calculate the histogram and normalize it
    hist_img = cv2.calcHist([img_hsv], [0,1], None, [360,256], [0,360,0,256])
    cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_img

def V_color_profile(image):

    image = cv2.imread(image)
    
    # convert it to V
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # calculate the histogram and normalize it
    hist_img = cv2.calcHist([img_hsv], [2], None, [256], [0,256])
    cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_img

def pca_reduction(hist_img):
    
    hist_img = HS_color_profile(hist_img)
    hist_img = hist_img.reshape(360, 256)

    pca = PCA(n_components=2)
    hist_pca = pca.fit_transform(hist_img)

    return hist_pca.flatten()

def create_color_vec(result, metric,test=False):
    color_vectors = []
    total_iterations = len(result) if not test else 1001  # Number of iterations based on 'test' flag

    # Use tqdm to display the progress
    with tqdm(total=total_iterations) as pbar:
        for index, row in result.iterrows():
            image_path = row['image_path']
            if metric == "hs":
                color_vector = pca_reduction(image_path)
                color_vector = HS_color_profile(image_path).flatten()
                color_vectors.append(color_vector)
            elif metric == "v":
                color_vector = V_color_profile(image_path).flatten()
                color_vectors.append(color_vector)
            else:
                print("ERROR: Invalit metric")

            pbar.update(1)  # Update the progress by 1
            if test and index == 1000:
                break

    return color_vectors

if __name__ == "__main__":
    import pandas as pd

    result = pd.read_csv("image_paths.csv")
    result

    #print(V_color_profile(result.loc[0, "image_path"]).shape)

    create_color_vec(result, metric="v",test=True)