# Author: Arturo Parrales Salinas
# Script with helper function for working with tensorflow certification prep
import itertools
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn import metrics
from typing import List
from typing import Dict
from typing import Optional


def plot_curves(history: object, keys: List[str] = None):
    """
    Plot loss and accuracy curves in separate plots.
    It uses history object from fitting a tensorflow model.

    Parameters
    ----------
    history: History
        History object retuned by Model.fit
    keys: List[str]
        List of string keys of the different curves stored in History.history
    """
    
    suffixes = ["loss", "accuracy"]
    names = [""]
    if keys:
        keys = [k+"_" for k in keys]
    names.extend(keys)

    epochs = range(1, len(history.history["loss"])+1)

    for suffix in suffixes:
        plt.figure()
        for name in names:
            curve = name + suffix
            plt.plot(epochs, history.history[curve], label=curve)
        plt.title(suffix)
        plt.xlabel("epochs")
        plt.legend();


def plot_random_images(
    images: np.ndarray,
    labels: List[int],
    n_samples: int = 5,
    samples_idx: Optional[List[int]] = None,
    labels_to_class_map: Optional[Dict[int, str]] = None,
):
    """
    Plot random number of 'n_samples' images with its respective label.

    Parameters
    ----------
    images: numpy.ndarray
        Matrix(tensor) of images' pixel values
    labels: List[Union[int, str]]
        Labels of the images. Must match the number of images
    n_samples: int
        Number of random images to plot
    samples_idx: Optional[List[int]]
        Indexes of the images in 'images' to plot. If specified it overrides 
        'n_samples' whic is the number of images to plot.
    labels_to_class_map: Optinal[Dict[int, str]]
        Dictionary that maps the labels to ac actual name of the class for the label.
    """
    if samples_idx is None:
        samples_idx = range(n_samples)
    else:
        n_samples = len(samples_idx)

    if labels_to_class_map is None:
        labels_to_class_map = dict()

    plt.figure()
    # adjust the subplots space 
    # hspace for space between rows
    plt.subplots_adjust(hspace=0.45)
    cols = 5
    rows = n_samples // cols
    rows += 0 if n_samples%cols == 0 else 1 
    for i, idx in enumerate(samples_idx):
        ax = plt.subplot(rows, cols, i+1)
        image, label = images[idx], labels[idx]
        # no need to use mpimg.imread() because the image is an np.array
        plt.imshow(image)
        label = labels_to_class_map.get(label, label)
        plt.title(f"{idx} - {label}", size=6)
        plt.axis("off")


def convert_preds(preds: np.ndarray, prediction_threshold: Optional[float] = 0.5):
  """ Rounds binary classification predictions to 0 or 1 based on threshold """
  rounded_preds = np.where(preds > prediction_threshold, 1, 0)
  return rounded_preds


def plot_confusion_matrix(
    y_true, y_pred, prediction_threshold=0.5, classes_labels=None
):

    y_preds = None

    if y_pred.shape[1] > 1 and y_true.shape[1] > 1:
        # multilabel
        raise NotImplementedError("No multilabel handling")
    elif y_pred.shape[1] > 1:
        # multiclass
        y_preds = tf.argmax(y_pred)
        print(y_pred[0])
        print(y_preds[0])
    elif y_pred.shape[1] == 1:
        # binary
        # round predictions
        y_preds = convert_preds(y_pred, prediction_threshold=prediction_threshold)
    else:
        raise ValueError("Unknown predictions type")  


    # generate confusion matrix
    cf_matrix = metrics.confusion_matrix(y_true, y_preds)

    # normalize confusion matrix - the percentage of data in each square
    # from the total data used to generate the matrix
    cfm_normalized = cf_matrix.astype("float") / cf_matrix.sum()

    
    # AVOID - SEEMS NOT THAT INSIGHTFUL FOR OVERALL SAMPLE
    # this is a weird normalization - it normalizes using the expected data in
    # positive, and false. Each category gets normalized against either positive
    # total samples or false total samples
    # bad_cfm = cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]

    # time to pretify in a plot
    fig, ax = plt.subplots()

    # create a matrix plot
    matrix_plot = ax.matshow(cf_matrix, cmap=plt.cm.Blues)
    # set gradient bar  (heat bar for total of samples)
    fig.colorbar(
        matrix_plot,
        # +1 to include all values in sample
        values=np.arange(0, cf_matrix.sum()+1), 
        label="Samples (max value is all samples)"
    )
    
    # create classes labels
    n_classes = cf_matrix.shape[0]
    if classes_labels:
        labels = classes_labels
    else:
        labels = np.arange(n_classes)
    
    # label axis
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    # set x axis to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # adjust label sizes
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    # set the color darkness/brightness
    gradient_threshold = (cf_matrix.max() + cf_matrix.min())/2.0

    # plot the text on each cell 
    # (itertools.product helps us make the matrix elements) 
    # if you have a=[1,2], b=[0,9], 
    # product(a,b) returns [(1,0), (1,9), (2,0), (2,9)] 
    for i, j in itertools.product(range(n_classes), range(cf_matrix.shape[1])):
        plt.text(
            j,
            i,
            f"{cf_matrix[i, j]} ({cfm_normalized[i, j]*100:.2f}%)",
            horizontalalignment="center",
            color="white" if cf_matrix[i, j] > gradient_threshold else "black",
            size=15, 
        )

    return cf_matrix, cfm_normalized


def calculate_confusion_matrix(
    y_true, y_pred, prediction_threshold=0.5, one_vs_all=False
):

    y_preds = None

    if y_pred.shape[1] > 1 and y_true.ndim > 1 and y_true.shape[1] > 1:
        # multilabel
        raise NotImplementedError("No multilabel handling")
    elif y_pred.shape[1] > 1:
        # multiclass
        # retrieve the index of the prediction in the sample with highest value,
        # that value must correspond to the true label (represented as a number/idx)
        y_preds = tf.argmax(y_pred, axis=1)
    elif y_pred.shape[1] == 1:
        # binary
        # round predictions
        y_preds = convert_preds(y_pred, prediction_threshold=prediction_threshold)
    else:
        raise ValueError("Unknown predictions type")  


    # generate confusion matrix
    cf_matrix = metrics.confusion_matrix(y_true, y_preds)

    if one_vs_all:
        # normalize using the expected data of one label vs the rest
        # Each category gets normalized separately against the
        # total samples aka expected label becomes positive, and all other n-1 
        # class labels become false (converting it to binary case)
        cfm_normalized = cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]
    else:
        # when no one_vs_all, calculate what percentage each prediction was 
        # correct in the overall dataset, so ...
        # normalize confusion matrix - the percentage of data in each square
        # from the total data used to generate the matrix
        cfm_normalized = cf_matrix.astype("float") / cf_matrix.sum()

    return cf_matrix, cfm_normalized


def plot_confusion_matrix(
    cf_matrix, cfm_normalized, classes_labels=None, text_size=5, figsize=(10,10)
):

    # time to pretify in a plot
    fig, ax = plt.subplots(figsize=figsize)

    # create a matrix plot
    matrix_plot = ax.matshow(cf_matrix, cmap=plt.cm.Blues)
    # set gradient bar  (heat bar for total of samples)
    fig.colorbar(
        matrix_plot,
        # +1 to include all values in sample
        values=np.arange(0, cf_matrix.sum()+1), 
        label="Samples (max value is all samples)"
    )
    
    # create classes labels
    n_classes = cf_matrix.shape[0]
    if classes_labels:
        labels = classes_labels
    else:
        labels = np.arange(n_classes)
    
    # label axis
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )

    # set x axis to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    # rotate xlabels in case the are long names
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )

    # adjust label sizes
    ax.yaxis.label.set_size(text_size*1.5)
    ax.xaxis.label.set_size(text_size*1.5)
    ax.title.set_size(text_size*1.5)

    # set the color darkness/brightness
    gradient_threshold = (cf_matrix.max() + cf_matrix.min())/2.0

    # plot the text on each cell 
    # (itertools.product helps us make the matrix elements) 
    # if you have a=[1,2], b=[0,9], 
    # product(a,b) returns [(1,0), (1,9), (2,0), (2,9)] 
    for i, j in itertools.product(range(n_classes), range(cf_matrix.shape[1])):
        plt.text(
            j,
            i,
            f"{cf_matrix[i, j]}\n({cfm_normalized[i, j]*100:.2f}%)",
            horizontalalignment="center",
            color="white" if cf_matrix[i, j] > gradient_threshold else "black",
            size=text_size, 
        )


def load_and_prep_image(filename, img_shape=224):
  """ 
  Reads img from filename, turns it into a tensor, reshapes it to
  (img_shape, img_shape, 3) and scales it between [0, 1]
  """
  # read img
  img = tf.io.read_file(filename)
  # decode read file into a tensor
  img = tf.image.decode_image(img)
  # resize
  img = tf.image.resize(img, size=[img_shape, img_shape])
  # rescale values between 0 and 1
  img = img/255.0
  return img


def unzip_folder(filename):
    """Unzips the downloaded file and saves it in the current dir"""
    with zipfile.ZipFile(filename) as zip_ref:
        zip_ref.extractall()


def walkthrough_dir(foldername):
    """Walk through the `filename` directory and list number of files and folders"""
    for dirpath, dirnames, filenames in os.walk(foldername):
        print(
            f"'{dirpath}' has {len(dirnames)} directories and {len(filenames)} files"
        )
