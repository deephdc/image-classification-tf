"""
Miscellaneous functions for test time.

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import numpy as np

from imgclas.data_utils import k_crop_data_sequence


def predict(model, X, conf, top_K=None, crop_num=10, filemode='local', merge=False, use_multiprocessing=False):
    """
    Predict function.

    Parameters
    ----------
    model: keras model instance
    X : str or list
        List of images paths of length N. If providing a list of urls, be sure to set correctly the 'filemode' parameter.
        If a str is provided it will be understood as a single image to predict.
    conf: dict
        Configuration parameters. The data augmentation parameters that will be used in the inference can be changed in
        conf['augmentation']['val_mode'].
    top_k : int
        Number of top predictions to return. If None, all predictions will be returned.
    crop_num: int
        Number of crops to use for test. Default is 10.
    filemode : str, {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.
    merge: Merge the predictions of all the images in the list. This value is tipically set to True when you pass
        multiple images of the same observation.
    use_multiprocessing: bool
       Use multiprocessing with the Keras generator.

    Returns
    -------
        pred_lab: np.array, shape (N, top_k)
            Array of predicted labels
        pred_prob:  np.array, shape (N, top_k)
            Array of predicted probabilities
    """

    if top_K is None:
        top_K = conf['model']['num_classes']
    if type(X) is str: #if not isinstance(X, list):
        X = [X]

    data_gen = k_crop_data_sequence(inputs=X,
                                    im_size=conf['model']['image_size'],
                                    mean_RGB=conf['dataset']['mean_RGB'],
                                    std_RGB=conf['dataset']['std_RGB'],
                                    preprocess_mode=conf['model']['preprocess_mode'],
                                    aug_params=conf['augmentation']['val_mode'],
                                    crop_mode='random',
                                    crop_number=crop_num,
                                    filemode=filemode)

    output = model.predict(data_gen,
                           verbose=1,
                           max_queue_size=10,
                           workers=4,
                           use_multiprocessing=use_multiprocessing)

    output = output.reshape(len(X), -1, output.shape[-1])  # reshape to (N, crop_number, num_classes)
    output = np.mean(output, axis=1)  # take the mean across the crops

    if merge:
        output = np.mean(output, axis=0)  # take the mean across the images
        lab = np.argsort(output)[::-1]  # sort labels in descending prob
        lab = lab[:top_K]  # keep only top_K labels
        lab = np.expand_dims(lab, axis=0)  # add extra dimension to make to output have a shape (1, top_k)
        prob = output[lab]
    else:
        lab = np.argsort(output, axis=1)[:, ::-1]  # sort labels in descending prob
        lab = lab[:, :top_K]  # keep only top_K labels
        prob = output[np.repeat(np.arange(len(lab)), lab.shape[1]),
                      lab.flatten()].reshape(lab.shape)  # retrieve corresponding probabilities

    return lab, prob


def topK_accuracy(true_lab, pred_lab, K=1):
    """
    Compute the top_K accuracy

    Parameters
    ----------
    true_lab: np.array, shape (N)
        Array with ground truth labels
    pred_lab: np.array, shape (N, M)
        Array with predicted labels. M should be bigger than K.
    K: int
        Accuracy type to compute
    """
    assert K<= pred_lab.shape[1], 'K is bigger than your number of predictions'
    mask = [lab in pred_lab[i, :K] for i, lab in enumerate(true_lab)]
    return np.mean(mask)
