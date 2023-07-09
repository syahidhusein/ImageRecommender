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

# Here we create a generator that uses uuid library to generate single unique hexadecimal IDs
def uuid_generator():
    while True:
        yield str(uuid.uuid4())

# Here we create a generator for every single row entry we want in the database
# Each row entry contains unique ID and image path
def paths_generator(init_directory,id_gen):
    for root, dirs, files in os.walk(init_directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                filepath = os.path.join(root,file)
                img = Image.open(filepath)
                if img.format == "GIF":
                    pass
                else:
                    yield next(id_gen),filepath

# generator class for creating entries
class main_generator:  
    def __init__(self, init_directory):
        id_gen = uuid_generator()
        self.gen = paths_generator(init_directory,id_gen)

    def gen_entry(self):
        return next(self.gen)

# just for testing functions
if __name__ == "__main__":
    path = "images"
    gen_row = main_generator(path)

    for i in range(5):
        print(gen_row.gen_entry())