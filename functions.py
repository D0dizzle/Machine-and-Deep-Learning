## File for the functions used in the Notebook
import os 
import numpy as np
from PIL import Image

def get_filenames_labels(folder):
    ''' Detects categories, filenames and labels of images
        Arguments
        ---------
        folder: string
            The folder where the images reside in subfolders. Subfolders correspond to the categories.

        Returns
        -------
        categories : list
            A list with all categories
        filenames: list
            A list of list of filenames of the images
        labels: list
            A list of list of the categories corresponding to filenames
    '''
    categories = [f.name for f in os.scandir(folder) if f.is_dir()]
    filenames = []
    labels = []
    for category in categories: 
        subfolder = os.path.join(folder, category)
        fnames = get_filenames(subfolder)
        filenames.append(fnames)
        labels.append([category] * len(fnames))

    return categories, flatten_lst(filenames), flatten_lst(labels)

def get_filenames(subfolder):
    imagefiles = [os.path.join(subfolder, filename) for filename in os.listdir(subfolder) if os.path.isdir(subfolder)] ## wo ist der Fehler?
    #imagefiles = [f for f in glob.glob(filepath+'*.jpg')]
    return imagefiles    

# Label list
def flatten_lst(lst):
    return [item for items in lst for item in items] # items = innere Liste in der äußeren Liste item = Element in innerer Liste


def read_images(filenames, height=None, width=None):
    images = [Image.open(filename) for filename in filenames]
    if (not height is None) and (not width is None):
        images = [img.resize((width, height)) for img in images]
    return images

def images_to_array(images):
    return np.asarray([np.asarray(img) for img in images])
