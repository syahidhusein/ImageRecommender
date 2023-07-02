import numpy as np
import pickle

def merge_pkls(pkl_name_list, out_name="cnn_embedding.pkl"):
    if not isinstance(pkl_name_list, list):
        raise ValueError("pkl_name_list has to be an list of .pkl files")
    # Create an empty list to store the combined data
    combined_data = []
    # Run through all files in the input directory
    for merge_pkl in pkl_name_list:
        with open(merge_pkl, "rb") as file:
            data = pickle.load(file)
            # Add the data to the combined list
            combined_data.extend(data)
    
    # Save the combined data in an output file
    with open(out_name, "wb") as file:
        pickle.dump(combined_data, file)

# Convert the embeddings list to a numpy array
def save_pkl(arr_vector, name):
    array = np.array(arr_vector)
    # Sverify that it is an array
    if not isinstance(array, np.ndarray):
        raise ValueError("arr_vector has to be an array")
    
    # Save the embeddings array to a file using pickle
    with open(name, 'wb') as file:
        pickle.dump(array, file)

# Load the saved embeddings array from the file
def load_pkl(name):
    with open(name, 'rb') as file:
        array = pickle.load(file)
        return array