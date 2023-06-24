from PIL import Image
import os
import uuid

## Creating generators
"""
Here we create the following generators
- reading the image
- creating a unique hexadecimal id for each image
- creates a row entry for each image
"""

#!!! not in use, exampel funktion !!!
# Here we create a generator that reads and displays the image,
# One for single image paths
def img_reader(image_path):
    with Image.open(image_path) as img:
        yield img

#!!! not in use, exampel funktion !!!
# Another for all images in a directory
def images_reader(folder_list):
    for category in folder_list:
        for file in os.listdir(path + "/" + category):
            with Image.open(path + "/" + category + "/" + file) as img:
                yield img

# Here we create a generator that uses uuid library to generate single unique hexadecimal IDs
def uuid_generator():
    while True:
        yield str(uuid.uuid4())

# Here we create a generator for every single row entry we want in the database
# Each row entry contains unique ID,image path, and image label
def row_generator(id_gen, folder_list, path):
    for category in folder_list:
        for file in os.listdir(path + "/" + category):
            filepath = path + "/" + category + "/" + file
            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                yield next(id_gen),filepath, category
            else:
                continue

def set_label(path):
    labels = os.listdir(path)
    labels.remove("metadata.txt")
    if ".DS_Store" in labels:
        labels.remove(".DS_Store")
    return labels

# generator class for creating entries
class main_generator:  
    def __init__(self, path):
        labels = set_label(path)
        id_gen = uuid_generator()
        self.gen = row_generator(id_gen, labels, path)

    def gen_entry(self):
        return next(self.gen)

# just for testing funktions
if __name__ == "__main__":
    path = "images/weather_image_recognition"
    gen_row = main_generator(path)

    for i in range(5):
        print(gen_row.gen_entry())