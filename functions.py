## File for the functions used in the Notebook
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

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

def minmax(a):
    a = np.asarray(a)
    a = (a - a.min()) / (a.max() - a.min())
    return a

def images_to_array(images):
    return np.asarray([np.asarray(img) for img in images])

def slice_data_in_folds(data, y, k):
    ''' Shuffles and splits the arrays of images and the array of labels in k-folds
        each folds contains a similar amount of elements
    Arguments
    ---------
    data: numpy.ndarray
        A 4D array of images shape: (num_samples, height, width, channels)
    y: numpy array
        A 2D array of the encoded labels shape: (num_samples, num_classes)
    k: int
        The number of folds 
    Returns
    -------
    image_folds : list of numpy.ndarray
        A list containing k arrays each with a set of images
    y_folds: list of numpy.ndarray
        A list containing k arrays each with a set of labels
    '''
    # Shuffle the data and labels in sync
    indices = np.arange(data.shape[0])  # Erzeuge einen Array der Indizes
    np.random.shuffle(indices)  # Shuffle die Indizes zufällig
    
    # Wende die zufällig sortierten Indizes auf die Daten und Labels an
    data = data[indices]
    y = y[indices]

    y_folds = np.array_split(y, k)
    image_folds = np.array_split(data, k)
    return image_folds, y_folds

def build_model(epochs, fold_x_train, fold_y_train, fold_x_val, fold_y_val, input_shape, y):
    ''' Builds and trains the model based on a specified amount of epochs.
        Uses fixed propagation, dropout and activation functions. 
        Evaluates the model after training and returns results and accuracy

        Arguments
        ---------
        epochs: int
            The number of epochs used to train the model.

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
    output = Dense(units=y.shape[-1], activation='softmax')(hidden)
    cnn = Model(inputs=inputs, outputs=output, name='CNN_CBP_Class')

    # Configuration of the training process
    cnn.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = cnn.fit(x=fold_x_train, y=fold_y_train, epochs=epochs, batch_size=32)
    # Fit model
    model_accuracy = cnn.evaluate(x=fold_x_val, y=fold_y_val, verbose=0)[1]
    return history, model_accuracy