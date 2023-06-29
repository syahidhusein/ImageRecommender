import numpy as np
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm

def HSV_color_profile(image):
    
    image = cv2.imread(image)
    
    #resized_img = cv2.resize(image, (128, 128))
    # convert it to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # calculate the histogram and normalize it
    hist_img = cv2.calcHist([img_hsv], [0,1], None, [360,256], [0,360,0,256])
    cv2.normalize(hist_img, hist_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return hist_img

def pca_reduction(hist_img):
    
    hist_img = HSV_color_profile(hist_img)
    hist_img = hist_img.reshape(360, 256)

    pca = PCA(n_components=2)
    hist_pca = pca.fit_transform(hist_img)

    return hist_pca.flatten()

def create_hsv_vec(result, test=False):
    hsv_vectors = []
    total_iterations = len(result) if not test else 1201  # Anzahl der Iterationen basierend auf 'test' Flag

    # Verwende tqdm, um den Fortschritt anzuzeigen
    with tqdm(total=total_iterations) as pbar:
        for index, row in result.iterrows():
            image_path = row['image_path']
            hsv_vector = pca_reduction(image_path)
            hsv_vector = HSV_color_profile(image_path).flatten()
            hsv_vectors.append(hsv_vector)
            pbar.update(1)  # Aktualisiere den Fortschritt um 1
            if test and index == 1200:
                break

    return hsv_vectors

if __name__ == "__main__":
    import pandas as pd

    result = pd.read_csv("image_paths.csv")
    result

    create_hsv_vec(result, test=True)