from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm
import numpy as np
import logging
from scripts import save_load

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

def create_color_vec(result, metric, batch_size=100, test=False, log_file=None, pkl_save=None):
    color_vectors = []
    # Determine the total number of batches
    total_batches= int(np.ceil(len(result)/batch_size))if not test else int(1001/batch_size)  # Number of iterations based on 'test' flag

    # Set up logging
    if log_file:
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Check if a previous state is logged
    start_from = 0
    if log_file:
        with open(log_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                last_iteration = int(last_line.split(" - ")[2].split(":")[1])
                start_from = last_iteration + 1

    # logging info 
    if start_from > 0:
        msg = f"__Continue processing from i={start_from}__"
        print(msg)
        logging.info(msg)

    # Use tqdm to display the progress
    for i in tqdm(range(start_from,total_batches),desc="Progress",unit="batch"):
        batch = result.iloc[i*batch_size:(i+1)*batch_size]  # Get the batch of rows from the dataframe
        img_paths = batch["image_path"].tolist()  # Get the image paths for the batch

        batch_vectors = []
        for img in img_paths:
            if metric == "hs":
                color_vector = pca_reduction(img)
                batch_vectors.append(color_vector)
            elif metric == "v":
                color_vector = V_color_profile(img).flatten()
                batch_vectors.append(color_vector)
            else:
                print("ERROR: Invalit metric")

        color_vectors.extend(batch_vectors)

        if test and i == 1000:
            break

        # Save progress in to a .pkl file
        if pkl_save:
            save_load.save_pkl(color_vectors, pkl_save)

        # Log the current iteration
        if log_file:
            logging.info("Current iteration: {}".format(i))

    return color_vectors

if __name__ == "__main__":
    import pandas as pd

    result = pd.read_csv("image_paths.csv")
    result

    #print(V_color_profile(result.loc[0, "image_path"]).shape)
    test = create_color_vec(result, metric="v",test=True)
    print(len(test[0]))