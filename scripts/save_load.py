import numpy as np
import pickle
import os

def merge_pkls(pkl_name_list, out_name="cnn_embedding.pkl"):
    if not isinstance(pkl_name_list, list):
        raise ValueError("pkl_name_list has to be an list of .pkl files")
    # Erstelle eine leere Liste zum Speichern der kombinierten Daten
    combined_data = []
    # Durchlaufe alle Dateien im Eingabeverzeichnis
    for merge_pkl in pkl_name_list:
        with open(merge_pkl, "rb") as file:
            data = pickle.load(file)
            
            # Füge die Daten zur kombinierten Liste hinzu
            combined_data.extend(data)
    
    # Speichere die kombinierten Daten in einer Ausgabedatei
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