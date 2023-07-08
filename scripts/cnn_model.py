from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging
from scripts import save_load

# Konfiguration der TensorFlow-GPU-Einstellungen
#tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
# check if gpu available
if tf.config.list_physical_devices('GPU'):
    print('GPU gefunden')
    
else:
    print('Keine GPU gefunden')

# Here we create arrays for each image that we will use for the similarity functions
class cnn_model():
    def __init__(self, name, pooling):
        self.name = name
        self.pooling = pooling

        if self.name == "resnet":
            base_model = ResNet50(weights='imagenet', include_top=False)
        elif self.name == "mobilenet":
            base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        else:
            raise ValueError(f'Invalid model metric "{self.name}"')

        if self.pooling == "maxpool":
            # For MaxPooling2D layer
            x = base_model.output
            x = MaxPooling2D(pool_size=(2, 2))(x)
            self.model = Model(inputs=base_model.input, outputs=x)
        elif self.pooling == "globavg":
            # For GlobalAveragePooling2D Layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            self.model = Model(inputs=base_model.input, outputs=x)
        else:
            raise ValueError(f'Invalid pooling metric "{self.pooling}"')

    def processing_img(self, image_path):
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img
    
    def extract_feature(self, img):
        features = self.model.predict(img)
        return features

## Generate embeddings in batches
# Here we try to generate vectors for images by batches
def create_cnn_embedding(results, model_name, pooling, batch_size=100, test=False, log_file=None, pkl_save=None):
    cnn_vectors = []
    model = cnn_model(name=model_name, pooling=pooling)

    # Determine the total number of batches
    total_batches = int(np.ceil(len(results) / batch_size)) if not test else int(1001/batch_size)

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

    # Iterate over the rows in the dataframe and extract vectors in batches
    for i in tqdm(range(start_from, total_batches), desc="Progress", unit="batch"):
        batch = results.iloc[i*batch_size:(i+1)*batch_size]  # Get the batch of rows from the dataframe
        img_paths = batch["image_path"].tolist()  # Get the image paths for the batch

        batch_vectors = []
        for img in img_paths:
            img_vector = model.processing_img(img)  # Extract vector
            batch_vectors.append(img_vector)

        batch_vectors = np.concatenate(batch_vectors, axis=0) #stacking the arrays among each other
        
        batch_vectors = model.extract_feature(batch_vectors)
        cnn_vectors.extend(batch_vectors)  # Add the batch vectors to the main list

        if test and (i+1) * batch_size >= 1000:
            break
        
        # Save progress in to a .pkl file
        if pkl_save:
            save_load.save_pkl(cnn_vectors, pkl_save)

        # Log the current iteration
        if log_file:
            logging.info("Current iteration: {}".format(i))

    return cnn_vectors

if __name__ == "__main__":
    #import os
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    import pandas as pd
    # If the CNN model is executed from here, import save_load must be commented out in the main script above in order to run it
    #import save_load
    results = pd.read_csv("image_paths.csv")

    cnn_test = create_cnn_embedding(results,
                                    model_name="mobilenet", 
                                    pooling="globavg", 
                                    batch_size=100,
                                    test=True, 
                                    log_file=None, 
                                    pkl_save=None)