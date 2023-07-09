import os
from PIL import Image
import pytest
from generator import paths_generator  # Replace "mymodule" with the actual module containing the function


def test_paths_generator():
    # Create a temporary directory and some sample image files
    directory = "unit_test_images"
    image1 = directory + "\image1.jpg"
    image2 = directory + "\image2.png"
    image3 = directory + "\image3.gif"
    image4 = directory + "\image4.jpg"

    # Define a dummy ID generator
    def dummy_id_gen():
        yield 1
        yield 2
        yield 3
        yield 4

    # Generate the expected results
    expected_results = [
        (1, str(image1)),
        (2, str(image2)),
    ]

    # Test the paths_generator function
    generator = paths_generator(str(directory), dummy_id_gen())
    results = list(generator)

    # Assert the expected results match the actual results
    assert results == expected_results
