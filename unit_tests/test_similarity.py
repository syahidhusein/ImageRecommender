import pytest
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm
from similarity import similarity_computation  # Replace "mymodule" with the actual module containing the function


@pytest.fixture
def input_vector():
    return [1, 2, 3]


@pytest.fixture
def reference_vectors():
    return [[1, 2, 3], [10, 2, 3], [1, 2, 10]]

def test_similarity_computation(input_vector, reference_vectors):
    df = similarity_computation(input_vector, reference_vectors, scoring_method=distance.euclidean)

    # Assert the type of the result
    assert isinstance(df, pd.DataFrame)

    # Assert the shape of the result
    assert df.shape == (3, 1)

    # Assert the expected result
    assert df["similarity_measure"].tolist() == [9,7,0]
