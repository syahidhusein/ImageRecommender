from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
#from sklearn.metrics import jaccard_score

#Here we create similarity functions and a dataframe to store the similarity measures

# Creating euclidean distance function
def euclidean_distance(v1, v2):
    return distance.euclidean(v1, v2)

# Creating cosine similarity function
def cosine_similarity(v1, v2):
    return 1 - distance.cosine(v1, v2)

# Creating manhattan distance function
def manhattan_distance(v1, v2):
    return distance.cityblock(v1, v2)

# Creating jaccard similarity function
# def jaccard_similarity(v1, v2):
#     return jaccard_score(v1, v2, average='weighted')

# Creating hamming distance function
def hamming_distance(v1, v2):
    return distance.hamming(v1, v2)

# Creating dataframe with similarity measures
def similarity_computation(input_vector,
                           reference_vectors,
                           scoring_method=euclidean_distance,
                          ):
    # If the scoring methods are distances the dataframe will be sorted ascendingly (closest to farthest)
    if scoring_method == euclidean_distance or scoring_method == manhattan_distance or scoring_method == hamming_distance:
        ascending=True
    # If the scoring methods are similarities the dataframe will be sorted descendingly (highest similarity to lowest)     
    else:
        ascending=False
        
    similarities = []
    # Use tqdm to display the progress
    with tqdm(total=len(reference_vectors)) as pbar:
        for v in reference_vectors:
            similarities.append(scoring_method(input_vector, v))
            pbar.update(1)  # Update the progress by 1

    return pd.DataFrame(similarities, columns=["similarity_measure"]).sort_values("similarity_measure",ascending=ascending)

if __name__ == "__main__":
    input_vector = [1, 2, 3]
    reference_vectors = [[1, 2, 3], [1, 2, 10], [10, 2, 3]]
    df = similarity_computation(input_vector, reference_vectors, scoring_method=distance.euclidean)
    print(df["similarity_measure"].tolist())
