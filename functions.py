## File for the functions used in the Notebook
import os
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

def get_filenames_labels(folder):
    ''' 
    Detects categories, filenames and labels of images
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
    ''' 
    Retrieves the filenames and paths of the image files in the specified subfolder

    Arguments
    --------
    subfolder : str
        The path to the subfolder whose contents should be listed.

    Returns
    -------
    imagefiles : list[str]
        The list containing the filenames (and the path) of all image files in the subfolder.
    '''
    imagefiles = [os.path.join(subfolder, filename) for filename in os.listdir(subfolder) if os.path.isdir(subfolder)]
    return imagefiles    

def flatten_lst(lst: List[List[Any]]) -> List[Any]:
    ''' 
    Flattens a two-dimensional list into a one-dimensional list.

    This function reduces the dimensionality of the given two-dimensional input list by
    extracting all elements of the inner lists into a single list.

    Arguments
    --------
    lst : list[list]
        A two-dimensional list.

    Returns
    -------
    flattened_list : list
        A one-dimensional list containing all elements of the input list.
    '''
    return [item for items in lst for item in items] # items = innere Liste in der äußeren Liste item = Element in innerer Liste

# height: Optional[int]=None bedeutet, dass height entweder ein Integer ist, oder bei None None bleibt!
def read_images(filenames: List[str], height: Optional[int]=None, width: Optional[int]=None) -> List: 
    ''' 
    Extracts the image data of each imagefile in the specified input list into a list of ImageFile objects.
    Optionally resizes the image data according to the specified height and width.

    Arguments
    --------
    filenames : list[str]
        A list containing the filenames (and paths) of images.

    height : int
        The height of the image in pixel.
    
    width : int
        The width of the image in pixel. 

    Returns
    -------
    images : list[ImageFile]
        A list containing the ImageFile objects for each element of the input list.
        
    '''
    images = [Image.open(filename) for filename in filenames]
    if (not height is None) and (not width is None):
        images = [img.resize((width, height)) for img in images]
    return images
#brauchen wir den?
def minmax(a):
    a = np.asarray(a)
    a = (a - a.min()) / (a.max() - a.min())
    return a

def images_to_array(images: List):
    ''' 
    Transforms the data of each ImageFile object in the input list into 4-dimensional numpy array.

    The resulting array has the following shape:
    (number_of_images, height, width, color_channels)

    Arguments
    --------
    images : list[ImageFile]
        A list containing the ImageFile objects for each element of the input list.

    Returns
    -------
    image_array : numpy.ndarray
        A 4-dimensional numpy array containing the RGB-values for each pixel of each image from the input list. 
        The shape of the numpy array is (number_of_images, height, width, color_channels):
        - number_of_images: the amount of images in the input list.
        - height: the height of each image in pixel.
        - width: the width of each image in pixel.
        - color_channels: the number of color channel (typically 3 for RGB).      
    '''
    return np.asarray([np.asarray(img) for img in images])

def label_to_array(labels: List[str]) -> np.ndarray:
    ''' 
    Converts a list of label strings into a numpy array.

    Arguments
    --------
    labels : list[str]
        A list containing the labels as strings.

    Returns
    -------
    label_array : numpy.ndarray
        A numpy array containing the labels from the input list.
    '''
    return np.asarray(labels)


def ownOneHotEncoder(labels, categories):
    ''' 
    Encodes the labels (strings) using One-Hot Encoding into a 2D numpy array.

    The output is a 2D array where each row represents a label, and each column corresponds
    to a category. For each label, the column of the corresponding category is set to 1, 
    and all other columns are set to 0.

    Arguments
    --------
    labels : numpy.NDArray
        A 1D array containing the labels (as string)

    categories : list
        A list containing all categories

    Returns
    -------
    encoded_labels : numpy.NDArray
        A 2D array containing the one hot encoded labels.
        
    '''    
    encoded_vectors = []
    for label in labels:
        vector = [1 if label == categorie else 0 for categorie in categories]
        encoded_vectors.append(vector)
    df = pd.DataFrame(encoded_vectors, columns=categories)
    encoded_labels = df.to_numpy()
    return encoded_labels

def slice_data_in_folds(data, y, k: int):
    ''' 
    Shuffles and splits the arrays of images and the array of labels in k-folds
    each folds contains a similar amount of elements, though if the number of elements 
    is not perfectly divisible by k, some folds may have one more element than others.

    Arguments
    --------
    data: numpy.ndarray
        A 4D array of images shape: (number_of_images, height, width, color_channels)
    y: numpy array
        A 2D array of the encoded labels shape: (number_of_labels, number_of_categories)
    k: int
        The number of folds 

    Returns
    --------
    image_folds : list[numpy.ndarray]
        A list containing k arrays each with a set of images (4D array).
    y_folds: list[numpy.ndarray]
        A list containing k arrays each with a set of labels (2D array).
    '''

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    data = data[indices]
    y = y[indices]

    y_folds = np.array_split(y, k)
    image_folds = np.array_split(data, k)
    return image_folds, y_folds

def stratified_k_fold(data, y, k):
    ''' 
    Splits the data into k-fold by dividing the data into the corresponding categories into a list of lists.

    Arguments
    --------
    data: numpy.ndarray
        A 4D array of images shape: (number_of_images, height, width, color_channels)
    y: numpy array
        A 2D array of the encoded labels shape: (number_of_labels, number_of_categories)
    k: int
        The number of folds 

    Returns
    --------
    image_folds : list[numpy.ndarray]
        A list containing k arrays each with a set of images (4D array).
    y_folds: list[numpy.ndarray]
        A list containing k arrays each with a set of labels (2D array).
    '''

    categories = np.unique(np.argmax(y, axis=1))
    categorie_lists = [[] for category in categories]
    for i, element in enumerate(data):
        category = np.argmax(y[i])
        categorie_lists[category].append(element)
        print(f"{i}. Länge der Kategorie-Listen")
        print(len(categorie_lists[category]))

    image_folds = [[] for fold in range(k)]
    label_folds = [[] for fold in range(k)]

    fold_index = 0  # Start beim ersten Fold
    for category in range(len(categorie_lists)):
        print(len(categorie_lists))
        for element in categorie_lists[category]:
            image_folds[fold_index].append(element)
            label_folds[fold_index].append(y[np.where(np.argmax(y, axis=1) == category)[0][0]])
            fold_index += 1 
            if fold_index == k:  # Wenn wir den letzten Fold erreicht haben, fange wieder bei 0 an
                fold_index = 0

    for i in range(k):
        image_folds[i] = images_to_array(image_folds[i])
        print(len(image_folds[i]))
        label_folds[i] = label_to_array(label_folds[i])
        print(len(label_folds[i]))

    return image_folds, label_folds

def build_model(epochs: int, fold_x_train, fold_y_train, fold_x_val, fold_y_val, input_shape: dict):
    ''' Builds and trains the model based on a specified amount of epochs.
        Uses fixed propagation, dropout and activation functions. 
        Evaluates the model after training and returns results and accuracy

        Arguments
        ---------
        epochs : int
            The number of epochs used to train the model.
        fold_x_train : numpy.ndarray
            A 4D array of images of k-1 folds for training the model
        fold_y_train : numpy.ndarray
            A 2D array of labels for the corresponding images of k-1 folds for training the model
        fold_x_val : numpy.ndarray
            A 4D array of images of a single fold for evaluating the model
        fold_y_val : numpy.ndarray
            A 2D array of labels for the corresponding images of a single fold for evaluating the model
        input_shape : Dict
            A dictionary containing the shape of the input (height, width, color_channels)

        Returns
        -------
        history : History object
            A dict containing the metrics 'loss' and 'accuracy' for each epoch

        model_accuracy: float
            A float containing the evaluated accuracy of the trained model
    '''
    # CNN model
    inputs = Input(shape=(input_shape.values()))
    hidden = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(inputs)
    hidden = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = MaxPooling2D(pool_size=(2,2))(hidden)
    hidden = Dropout(rate=0.25)(hidden)
    hidden = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = MaxPooling2D(pool_size=(2,2))(hidden)
    hidden = Dropout(rate=0.25)(hidden)
    hidden = Flatten()(hidden)
    hidden = Dense(units=256, activation='relu')(hidden)
    hidden = Dropout(rate=0.25)(hidden)
    output = Dense(units=fold_y_train.shape[-1], activation='softmax')(hidden)
    cnn = Model(inputs=inputs, outputs=output, name='CNN_CBP_Class')

    # Configuration of the training process
    cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = cnn.fit(x=fold_x_train, y=fold_y_train, epochs=epochs, batch_size=32)
    # Fit model
    model_accuracy = cnn.evaluate(x=fold_x_val, y=fold_y_val, verbose=0)[1]
    return history, model_accuracy