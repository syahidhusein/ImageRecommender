import sqlite3
import pandas as pd
import numpy as np
import os

# import of own functions
from scripts import save_load, plot_funktion, similarity, HSV, cnn_model

def unknown_paths(folder_path):
    file_paths = []

    # Go through the files in the folder and extract the file paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    # Create a DataFrame with the file paths
    df = pd.DataFrame({"image_path": file_paths})

    return df

def process_data(df_unseen, model="mobilenet"):
    ## Create embeddings for unseen
    cnn_unknown = cnn_model.create_cnn_embedding(df_unseen, 
                                                model_name=model, 
                                                pooling="globavg",
                                                batch_size=10, 
                                                test=False, 
                                                log_file=None, 
                                                pkl_save=None
                                                )

    hs_unknown = HSV.create_color_vec(df_unseen, metric="hs", batch_size=10, test=False, log_file=None, pkl_save=None)
    v_unknown = HSV.create_color_vec(df_unseen, metric="v", batch_size=10, test=False, log_file=None, pkl_save=None)
    
    return cnn_unknown, hs_unknown, v_unknown

def img_similarity_recomender(_cnn, 
                              _hs, 
                              _v, 
                              _df, 
                              index=0, 
                              model="mobilenet", 
                              scoring_method=similarity.cosine_similarity
                              ):
    
    # Connects to .db file with SQLite
    conn = sqlite3.connect("databases/image_database.db")

    # Create a dataframe with values from image_paths table
    sql_query = "SELECT * FROM image_paths"
    results = pd.read_sql(sql_query, con=conn)
    results = pd.concat([_df, results], ignore_index=True)

    # Load all created cnn embeddings
    cnn_embeddings = save_load.load_pkl(f"cnn_embedding_{model}.pkl")
    # Load all created embeddings
    hs_vectors = save_load.load_pkl("hs_color_vector.pkl")
    # Load all created embeddings
    v_vectors = save_load.load_pkl("v_color_vector.pkl")

    ## concatinate new array on top
    cnn_embeddings = np.concatenate((_cnn,cnn_embeddings), axis=0)
    hs_vectors = np.concatenate((_hs,hs_vectors), axis=0)
    v_vectors = np.concatenate((_v,v_vectors), axis=0)

    ## similarity calculation cnn
    cnn_sim_df = similarity.similarity_computation(_cnn[index], cnn_embeddings, scoring_method=scoring_method)
    index_values = cnn_sim_df.index.tolist()

    column_values = results.loc[index_values, "image_path"]
    cnn_df_sim_path = pd.DataFrame(column_values)

    ## similarity calculation hs
    hs_sim_df = similarity.similarity_computation(_hs[index], hs_vectors, scoring_method=scoring_method)
    index_values = hs_sim_df.index.tolist()

    column_values = results.loc[index_values, "image_path"]
    hs_df_sim_path = pd.DataFrame(column_values)

    ## similarity calculation v
    v_sim_df = similarity.similarity_computation(_v[index], v_vectors, scoring_method=scoring_method)
    index_values = v_sim_df.index.tolist()

    column_values = results.loc[index_values, "image_path"]
    v_df_sim_path = pd.DataFrame(column_values)

    ## plot cnn
    top_8_paths, top_8_score = plot_funktion.top_8_gen(cnn_df_sim_path, cnn_sim_df)
    plot_funktion.images_grid(top_8_paths, top_8_score, title=f"{model} CNN embeddings similarity")

    ## plot hs
    top_8_paths, top_8_score = plot_funktion.top_8_gen(hs_df_sim_path, hs_sim_df)
    plot_funktion.images_grid(top_8_paths, top_8_score, title="Hue & Saturation (HSV) color similarity")
    plot_funktion.images_histogram_grid(top_8_paths, top_8_score, V=False)

    ## plot v
    top_8_paths, top_8_score = plot_funktion.top_8_gen(v_df_sim_path, v_sim_df)
    plot_funktion.images_grid(top_8_paths, top_8_score, title="Value (HSV) color similarity")
    plot_funktion.images_histogram_grid(top_8_paths, top_8_score, H=False, S=False)
