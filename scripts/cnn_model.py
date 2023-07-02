from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
import numpy as np
from tqdm import tqdm
import logging
from scripts import save_load

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

    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(128, 128))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = self.model.predict(img)
        return features.flatten()

### !!!!! Just for testing !!!!!
## Generate all together at once
# Here we try to generate vectors for all images at once (not efficient!!)
def __embedding_test__(results, model_name, pooling, test=False, log_file=None):
    cnn_vectors = []
    model = cnn_model(name=model_name, pooling=pooling)
    
    total_iterations = len(results) if not test else 1201  # Number of iterations based on the 'test' flag

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

    # iterate over the rows in the dataframe and extract cnn embeddings
    for index, row in tqdm(results.iterrows(), total=total_iterations, desc="Progress", initial=start_from):
        if index < start_from:
            continue

        img_path = row["image_path"]
        cnn_vector = model.extract_features(img_path)
        cnn_vectors.append(cnn_vector)
        
        if test and index == 1200:
            break

        # Log the current iteration
        if log_file:
            logging.info("Current iteration: {}".format(index))

    return cnn_vectors

## Generate in batches
# Here we try to generate vectors for images by batches
def create_cnn_embedding(results, model_name, pooling, test=False, log_file=None, pkl_save=None):
    batch_size = 50
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
            img_vector = model.extract_features(img)  # Extract vector
            batch_vectors.append(img_vector)

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
    import pandas as pd
    import psutil
    # If the CNN model is executed from here, import save_load must be commented out in the main script above
    import save_load
    results = pd.read_csv("image_paths.csv")

    # Get memory usage in bytes
    memory_usage = psutil.virtual_memory().used
    # Convert memory usage to human-readable format
    memory_usage_readable = psutil.virtual_memory().used / (1024 ** 2)  # Convert to megabytes

    cnn_test = create_cnn_embedding(results,
                                    model_name="mobilenet", 
                                    pooling="globavg", 
                                    test=True, 
                                    log_file="logfile.log", 
                                    pkl_save="cnn_embedding.pkl")

    # Get memory usage in bytes
    memory_usage_altogether = psutil.virtual_memory().used - memory_usage
    print(f"Memory Usage when generating altogether: {memory_usage_altogether} bytes")
    # Convert memory usage to human-readable format
    memory_usage_readable_altogether = psutil.virtual_memory().used / (1024 ** 2) - memory_usage_readable  # Convert to megabytes
    print(f"Readable Memory Usage when generating altogether: {memory_usage_readable_altogether} MB")

    print("test the length of vectors")
    print(len(results))
    print(len(cnn_test))