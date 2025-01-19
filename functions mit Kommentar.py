## File for the functions used in the Notebook

# Beschreibung: Dieses Modul bietet Funktionen zur Interaktion mit dem Betriebssystem.
# Nutzen im Code:
# Zugriff auf Dateisysteme (z. B. os.listdir() listet Dateien in einem Verzeichnis auf).
# Navigation in Ordnerstrukturen (z. B. os.path.join() erstellt Dateipfade).
import os


# Beschreibung: Typ-Hinweise, die den Code verständlicher machen und Fehler vermeiden helfen.
# Nutzen im Code:
# List: Gibt an, dass eine Funktion eine Liste erwartet oder zurückgibt.
# Optional: Zeigt an, dass ein Argument optional ist (z. B. height: Optional[int]).
# Any: Platzhalter für beliebige Datentypen.
from typing import Any, List, Optional


# Beschreibung: NumPy ist die wichtigste Bibliothek für numerische Berechnungen in Python.
# Nutzen im Code:
# Arrays für numerische Daten (z. B. Bilddaten).
# Mathematische Funktionen (z. B. Rotation, Mittelwertberechnung).
import numpy as np


# Beschreibung: Pandas ist eine Datenanalysebibliothek.
# Nutzen im Code:
# Erstellung von Tabellen (DataFrames).
# Konvertieren von Daten (z. B. Labels in One-Hot-Encoded Vektoren).
import pandas as pd


# Beschreibung: Pillow ist eine Bildverarbeitungsbibliothek.
# Nutzen im Code:
# Laden und Bearbeiten von Bildern.
from PIL import Image


# TensorFlow ist ein Framework für maschinelles Lernen, das oft mit Keras kombiniert wird, einer High-Level-API für neuronale Netze.
import tensorflow as tf


# Model: Ermöglicht die Erstellung eines neuronalen Netzwerks durch Definition von Eingabe- und Ausgabeschichten.
from tensorflow.keras.models import Model


# Beschreibung: Stellt verschiedene Schichten (Layers) für neuronale Netze bereit.
# Genutzte Layers:
# Input: Definiert die Eingabe des Modells.
# Conv2D: Führt die Faltung auf Bilder durch, um Merkmale zu extrahieren.
# MaxPooling2D: Reduziert die Größe der Feature Maps.
# Dropout: Verhindert Überanpassung, indem es zufällig Neuronen während des Trainings deaktiviert.
# Flatten: Wandelt 2D-Daten in 1D um.
# Dense: Vollständig verbundene Schicht (klassifiziert die Merkmale).
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten


# EarlyStopping: Stoppt das Training frühzeitig, wenn die Leistung auf den Validierungsdaten nicht mehr besser wird.
from tensorflow.keras.callbacks import EarlyStopping


# RMSprop: Ein Optimierungsalgorithmus, der die Lernrate dynamisch anpasst.
from tensorflow.keras.optimizers import RMSprop

#### Extracting and Formatting input data ####
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
    # Liest alle Unterordner (is_dir()) aus einem angegebenen Hauptordner (folder).
    # Jeder Unterordner stellt eine Kategorie dar.
    categories = [f.name for f in os.scandir(folder) if f.is_dir()]
    filenames = []
    labels = []

    # Für jede Kategorie wird der entsprechende Unterordner durchsucht (subfolder), und die Dateinamen (fnames) werden gesammelt.
    # Gleichzeitig wird eine Liste der Labels erstellt, wobei jedes Bild der aktuellen Kategorie zugeordnet wird ([category] * len(fnames)).
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
    # Baut vollständige Dateipfade (os.path.join) für alle Dateien in einem Unterordner.
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
    # Wandelt eine verschachtelte Liste in eine flache Liste um.
    return [item for items in lst for item in items]

# Lädt Bilder und skaliert sie.
# Lädt die Bilder mit PIL.Image und passt sie an die vorgegebene Höhe und Breite an.
# Rückgabe: Eine Liste von PIL.Image-Objekten.
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

#### Transforming lists into numpy arrays ####
# Wandelt eine Liste von Bildern in ein Numpy-Array um.
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
    # Jedes Bild wird in ein Array von Pixelwerten umgewandelt.
    # Rückgabe: Ein 4D-Array mit der Form (Anzahl_Bilder, Höhe, Breite, Farbkanäle).
    return np.asarray([np.asarray(img) for img in images])

# Wandelt Labels in ein Numpy-Array um. (['lion', 'tiger'])
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

# Führt Datenaugmentation durch.

def augment_images(images, labels):
    ''' 
    Augments the images using numpy.

    Arguments
    --------
    images : numpy.ndarray
        A numpy array containing the original images.
    labels : numpy.ndarray
        A numpy array containing the labels of the original images.

    Returns
    -------
    aug_images : list
        A numpy array containing the original and augmentated images.
    aug_labels : list
        A numpy array containing the labels of the original and augmentated images.
    '''
    aug_images = []
    aug_labels = []
    for i, row in enumerate(images):
        # Original
        aug_images.append(images[i])
        aug_labels.append(labels[i])
        
        # 90-degree rotation (to the right)
        aug_images.append(np.rot90(row, k=1))
        aug_labels.append(labels[i])
        
        # 180-degree rotation (upside down)
        aug_images.append(np.rot90(row, k=2))
        aug_labels.append(labels[i])
        
        # 270-degree rotation (to the left)
        aug_images.append(np.rot90(row, k=3))
        aug_labels.append(labels[i])
    
    # Augmentiert jedes Bild durch Rotationen (90°, 180°, 270°). 
    # Ergänzt die Labels entsprechend.
    # Rückgabe: 
    # aug_images: Liste der Original- und augmentierten Bilder.
    # aug_labels: Labels für alle augmentierten Bilder.

    return aug_images, aug_labels

#### Encoding the labels with OneHotEncoding ####
# Wandelt Labels in One-Hot-Encoded Vektoren um.
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
# Für jedes Label wird ein Vektor erstellt, z. B.:
# Kategorien: [lion, tiger]
# Label: lion
# Ausgabe: [0, 1]

#### Functions for k-fold cross validation ####
# Teilt die Daten in k gleich große Folds.
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
    # die indices sind ein array von 0 bis "Anzahl Einträge in data -1"
    # genauer: data.shape[0] ist die Anzahl an Zeilen in dem Array "data"
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices) # mischen
    #data wird mit den gemischten Indices neu erstellt
    data = data[indices]
    y = y[indices]

    y_folds = np.array_split(y, k)
    image_folds = np.array_split(data, k)
    return image_folds, y_folds

# Die Daten und Labels werden zufällig gemischt (np.random.shuffle) und in k Teile aufgeteilt (np.array_split).
# Beispiel: data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# indices = [0, 1, 2]
# nach dem shuffle: indices = [1, 0, 2] -> np.random.shuffle ist in-place und muss nicht in Variable gespeichert werden
# data = data[indices] -> data = [[4, 5, 6], [1, 2, 3], [7, 8, 9]]
# -> gleiches mit y :)

# Teilt die Daten in Folds mit gleichmäßiger Verteilung der Kategorien.
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
    # np.argmax(y, axis=1) -> gibt für jeden Eintrag in y an, an welchem Index der Wert maximal ist
    # Beispiel: y = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    # np.argmax(y, axis=1) = [0, 1, 0, 3] -> für [1, 0, 0, 0] Index = 0 für [0, 0, 0, 1] Index = 3
    # np.unqiue(np.argmax(y, axis=1)) -> [0, 1, 3], zweiter 0-Wert wird ignoriert, da nur jeder unique Wert genommen wird
    categories = np.unique(np.argmax(y, axis=1))
    categorie_lists = [[] for category in categories]
    
    # category für 4 Kategorien ist = [0, 1, 2, 3]
    # für i in allen data-Einträgen wird die Kategorie ermittelt und der Index des Eintrags aus data
    # in die richtige "sub-Liste" gespeichert
    # Beispiel: y[i] = [0, 1, 0, 0] -> np.argmax(y[i]) = 1 -> categorie_lists[1].append(Index des Data-Eintrags)
    for i in range(len(data)):
        category = np.argmax(y[i])  
        categorie_lists[category].append(i)
    
    # Jede Kategorie wird separat in eine Liste gepackt.
    # jetzt wird jede Kategorie in sich geshuffelt:
    # Für jede "sub-Liste" in categorie_lists:
    # 1. Erstellen wir ein Array von Indizes (0 bis Länge der Sub-Liste - 1),
    # 2. Diese Indizes werden gemischt (np.random.shuffle),
    # 3. Danach wird die "sub-Liste" in der Reihenfolge der geshuffelten Indizes neu befüllt.
    #
    # Beispiel: categorie_lists[0] = [0, 4, 6, 12] -> indices = [0, 1, 2, 3]
    # np.random.shuffle(indices) = [1, 3, 2, 0]
    # categorie_lists[0] = [categorie_lists[0][i] for i in [1, 3, 2, 0]] 
    # -> [categorie_lists[0][1], categorie_lists[0][3], categorie_lists[0][2], categorie_lists[0][0]] 
    # -> [4, 12, 6, 0]
    for category in range(len(categorie_lists)):
        indices = np.arange(len(categorie_lists[category]))
        np.random.shuffle(indices)
        categorie_lists[category] = [categorie_lists[category][i] for i in indices] ### wichtig!! i ist KEIN Iterator, sondern der Wert im indices Array!

    # Anschließend werden die Indizes der Bilder auf die Folds verteilt.
    image_folds = [[] for fold in range(k)]
    label_folds = [[] for fold in range(k)]

    # categorie_lists[category] enthält die Indizes für die Elemente in Data und y 
    # Verteilung der Datenpunkte auf die Folds:
    # Mehrere Iterationen:
    # 1. Über alle Kategorien in categorie_lists iterieren -> jede Kategorie wird nacheinander betrachtet.
    # 2. Für jeden Index in der jeweiligen Kategorie:
    #    a. Das Bild und Label, die zu diesem Index gehören, werden dem aktuellen Fold hinzugefügt.
    #    b. Der fold_index wird um 1 erhöht, um den nächsten Fold vorzubereiten.
    #    c. Wenn fold_index == k, bedeutet das, dass alle Folds einmal durchlaufen wurden. 
    #       -> Wir setzen fold_index zurück auf 0, um wieder beim ersten Fold weiterzumachen.
    # 3. Dadurch wird sichergestellt, dass die Daten gleichmäßig auf alle Folds verteilt werden, 
    #    auch wenn eine Kategorie mehr Datenpunkte hat als die anderen.
    fold_index = 0  # Start beim ersten Fold
    for category in range(len(categorie_lists)):
        for index in categorie_lists[category]:
            image_folds[fold_index].append(data[index])
            label_folds[fold_index].append(y[index])
            fold_index += 1 
            if fold_index == k:  # Wenn wir den letzten Fold erreicht haben, fange wieder bei 0 an
                fold_index = 0
    # für jeden Fold werden die "Sub-Liste" in ein Array umgewandelt
    for i in range(k):
        image_folds[i] = images_to_array(image_folds[i])
        label_folds[i] = label_to_array(label_folds[i])

    return image_folds, label_folds

#### Function to build and train the model ####
# Erstellt und trainiert ein CNN.
def build_model(epochs: int, fold_x_train, fold_y_train, fold_x_val, fold_y_val, fold_x_test, fold_y_test, input_shape: dict):
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

    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
    )
    optimizer = RMSprop(learning_rate=0.0001)
    # CNN model
    # Convolutional Layers: Extrahieren Merkmale.
    # Pooling Layers: Reduzieren die räumliche Größe.
    # Dropout: Verhindert Überanpassung.
    # Fully Connected Layer: Klassifiziert die Bilder.
    inputs = Input(shape=(input_shape.values()))
    hidden = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(inputs)
    hidden = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = MaxPooling2D(pool_size=(2,2))(hidden)
    hidden = Dropout(rate=0.25)(hidden)
    hidden = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = MaxPooling2D(pool_size=(2,2))(hidden)
    hidden = Dropout(rate=0.3)(hidden)
    hidden = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu')(hidden)
    hidden = MaxPooling2D(pool_size=(2,2))(hidden)
    hidden = Dropout(rate=0.35)(hidden)
    hidden = Flatten()(hidden)
    hidden = Dense(units=256, activation='relu')(hidden)
    hidden = Dropout(rate=0.25)(hidden)
    output = Dense(units=fold_y_train.shape[-1], activation='softmax')(hidden)
    cnn = Model(inputs=inputs, outputs=output, name='CNN_CBP_Class')

    # Configuration of the training process
    cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = cnn.fit(x=fold_x_train, y=fold_y_train,validation_data=(fold_x_val, fold_y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])
    # Fit model
    model_accuracy = cnn.evaluate(x=fold_x_test, y=fold_y_test, verbose=0)[1]
    return history, model_accuracy, cnn

#### Calculating the accuracy of the prediction ####
# Berechnet die Genauigkeit.
def accuracy(actuals, preds):
    '''
    Calculates the accuracy by comparing the actual categories with the predicted categories.

    Arguments
    --------
    actuals: numpy.ndarray
        A 1D array containing the indices of the actual category in a OneHotEncoded vector.
    preds: numpy.ndarray
        A 1D array containing the indices of the predicted category.

    Returns
    --------
    accuracy : float
        The proportion of correct predictions. For each match of "actuals" and "preds" counts as 1, each mismatch counts as 0.
        The result is the mean value of these comparisons.
    '''
    actuals, preds = np.asarray(actuals), np.asarray(preds)
    return np.mean(np.ravel(actuals) == np.ravel(preds))

# Vergleicht die tatsächlichen Labels (actuals) mit den vorhergesagten (preds).
# Rückgabe: Anteil korrekt klassifizierter Bilder.