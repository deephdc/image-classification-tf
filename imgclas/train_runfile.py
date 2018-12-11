"""
Training runfile

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Description:
This file contains the commands for training a convolutional net for image classification.

Additional notes:
* On the training routine: Preliminary tests show that using a custom lr multiplier for the lower layers yield to better
results than freezing them at the beginning and unfreezing them after a few epochs like it is suggested in the Keras
tutorials.
"""

#TODO List:

#TODO: Implement resuming training
#TODO: Try that everything works out with validation data
#TODO: Try several regularization parameters
#TODO: Add additional metrics for test time in addition to accuracy

import os
import time
import json
from datetime import datetime

import numpy as np

from imgclas.data_utils import load_data_splits, compute_meanRGB, compute_classweights, load_class_names, data_sequence
from imgclas import paths, config, model_utils, utils
from imgclas.optimizers import customAdam


def train_fn(TIMESTAMP, CONF):

    paths.timestamp = TIMESTAMP

    utils.create_dir_tree()
    utils.backup_splits()

    # Load the training data
    X_train, y_train = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                                        im_dir=CONF['general']['images_directory'],
                                        split_name='train')

    # Load the validation data
    if (CONF['training']['use_validation']) and ('val.txt' in os.listdir(paths.get_splits_dir())):
        X_val, y_val = load_data_splits(splits_dir=paths.get_ts_splits_dir(),
                                        im_dir=CONF['general']['images_directory'],
                                        split_name='val')
    else:
        print('No validation data.')
        X_val, y_val = None, None
        CONF['training']['use_validation'] = False

    # Load the class names
    class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())

    # Update the configuration
    CONF['model']['preprocess_mode'] = model_utils.model_modes[CONF['model']['modelname']]
    if CONF['model']['num_classes'] is None:
        CONF['model']['num_classes'] = len(class_names)

    assert CONF['model']['num_classes'] >= np.amax(y_train), "Your train.txt file has more categories than those defined in classes.txt"
    if CONF['training']['use_validation']:
        assert CONF['model']['num_classes'] >= np.amax(y_val), "Your val.txt file has more categories than those defined in classes.txt"

    # Compute the class weights
    if CONF['training']['class_weights_mode'] is not None:
        class_weights = compute_classweights(y_train,
                                             max_dim=CONF['model']['num_classes'],
                                             mode=CONF['training']['class_weights_mode'])
    else:
        class_weights = None

    # Compute the mean and std RGB values
    if CONF['dataset']['mean_RGB'] is None:
        CONF['dataset']['mean_RGB'], CONF['dataset']['std_RGB'] = compute_meanRGB(X_train)

    #Create data generator for train and val sets
    train_gen = data_sequence(X_train, y_train,
                              batch_size=CONF['training']['batch_size'],
                              num_classes=CONF['model']['num_classes'],
                              im_size=CONF['model']['image_size'],
                              mean_RGB=CONF['dataset']['mean_RGB'],
                              std_RGB=CONF['dataset']['std_RGB'],
                              preprocess_mode=CONF['model']['preprocess_mode'],
                              aug_params=CONF['augmentation']['train_mode'])
    train_steps = int(np.ceil(len(X_train)/CONF['training']['batch_size']))

    if CONF['training']['use_validation']:
        val_gen = data_sequence(X_val, y_val,
                                batch_size=CONF['training']['batch_size'],
                                num_classes=CONF['model']['num_classes'],
                                im_size=CONF['model']['image_size'],
                                mean_RGB=CONF['dataset']['mean_RGB'],
                                std_RGB=CONF['dataset']['std_RGB'],
                                preprocess_mode=CONF['model']['preprocess_mode'],
                                aug_params=CONF['augmentation']['val_mode'])
        val_steps = int(np.ceil(len(X_val)/CONF['training']['batch_size']))
    else:
        val_gen = None
        val_steps = None

    # Launch the training
    t0 = time.time()

    # Create the model and compile it
    model, base_model = model_utils.create_model(CONF)

    # Get a list of the top layer variables that should not be applied a lr_multiplier
    base_vars = [var.name for var in base_model.trainable_variables]
    all_vars = [var.name for var in model.trainable_variables]
    top_vars = set(all_vars) - set(base_vars)
    top_vars = list(top_vars)

    # # Set trainable variables
    # for layer in base_model.layers:
    #     layer.trainable = True

    model.compile(optimizer=customAdam(lr=CONF['training']['initial_lr'],
                                       amsgrad=True,
                                       lr_mult=0.1,
                                       excluded_vars=top_vars
                                       ),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(generator=train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=CONF['training']['epochs'],
                                  class_weight=class_weights,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=utils.get_callbacks(CONF),
                                  verbose=1, max_queue_size=10, workers=4,
                                  use_multiprocessing=True, initial_epoch=0)

    # Saving everything
    print('Saving data to {} folder.'.format(paths.get_timestamped_dir()))
    print('Saving training stats ...')
    stats = {'epoch': history.epoch,
             'training time (s)': round(time.time()-t0, 2)}
    stats.update(history.history)
    stats_dir = paths.get_stats_dir()
    with open(os.path.join(stats_dir, 'stats.json'), 'w') as outfile:
        json.dump(stats, outfile, sort_keys=True, indent=4)

    print('Saving the configuration ...')
    model_utils.save_conf(CONF)

    print('Saving the model to h5...')
    fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.h5')
    model.save(fpath)

    # print('Saving the model to protobuf...')
    # fpath = os.path.join(paths.get_checkpoints_dir(), 'final_model.proto')
    # model_utils.save_to_pb(model, fpath)

    print('Finished')
    return paths.get_timestamped_dir()

if __name__ == '__main__':

    CONF = config.conf_dict()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(' ', '_')

    train_fn(TIMESTAMP=timestamp, CONF=CONF)
