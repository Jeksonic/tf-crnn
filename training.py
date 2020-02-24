#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import logging
logging.getLogger("tensorflow").setLevel(logging.INFO)

from tf_crnn.config import Params
from tf_crnn.model import get_model_train
from tf_crnn.preprocessing import data_preprocessing
from tf_crnn.data_handler import dataset_generator
from tf_crnn.callbacks import CustomLoaderCallback, CustomSavingCallback, LRTensorBoard
from tf_crnn.config import CONST
import tensorflow as tf
import numpy as np
import os
import json
import shutil
import pickle
import click
from glob import glob
from sacred import Experiment, SETTINGS

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

ex = Experiment('crnn')


@ex.main
def training(_config: dict):
    parameters = Params(**_config)

    export_config_filename = os.path.join(parameters.output_model_dir, CONST.CONFIG_FILENAME)
    saving_dir = os.path.join(parameters.output_model_dir, CONST.FOLDER_SAVED_MODEL)

    is_dir_exist = os.path.isdir(parameters.output_model_dir)
    is_dir_del = parameters.del_output_model_dir
    is_dir_restore = parameters.restore_model
    if not is_dir_exist:
        is_dir_restore = False
        os.makedirs(parameters.output_model_dir)
    elif is_dir_del:
        is_dir_restore = False
        shutil.rmtree(parameters.output_model_dir)
        os.makedirs(parameters.output_model_dir)
    elif not is_dir_restore:
        assert not is_dir_exist, \
            '{} already exists, you cannot use it as output directory.'.format(parameters.output_model_dir)
        os.makedirs(parameters.output_model_dir)

    # data and csv pre-processing
    csv_train_file, csv_eval_file, \
        n_samples_train, n_samples_eval = data_preprocessing(parameters)

    parameters.train_batch_size = min(parameters.train_batch_size, n_samples_train)
    parameters.eval_batch_size = min(parameters.eval_batch_size, n_samples_eval)

    # export config file in model output dir
    with open(export_config_filename, 'w') as file:
        json.dump(parameters.to_dict(), file)

    # Create callbacks
    log_dir = os.path.join(parameters.output_model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 profile_batch=0)

    lrtb_callback = LRTensorBoard(log_dir=log_dir,
                                  profile_batch=0)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                       patience=10,
                                                       cooldown=0,
                                                       min_lr=1e-8,
                                                       verbose=1)

    es_callback = tf.keras.callbacks.EarlyStopping(min_delta=0.005,
                                                   patience=20,
                                                   verbose=1)

    sv_callback = CustomSavingCallback(saving_dir,
                                       saving_freq=parameters.save_interval,
                                       save_best_only=True,
                                       keep_max_models=3)

    list_callbacks = [tb_callback, lrtb_callback, lr_callback, es_callback, sv_callback]

    if is_dir_restore:
        last_time_stamp = max([int(p.split(os.path.sep)[-1].split('-')[0])
                               for p in glob(os.path.join(saving_dir, '*'))])

        loading_dir = os.path.join(saving_dir, str(last_time_stamp))
        ld_callback = CustomLoaderCallback(loading_dir)

        list_callbacks.append(ld_callback)

        with open(os.path.join(loading_dir, CONST.EPOCH_FILENAME), 'rb') as f:
            initial_epoch = pickle.load(f)

        epochs = initial_epoch + parameters.n_epochs
    else:
        initial_epoch = 0
        epochs = parameters.n_epochs

    # Get model
    model = get_model_train(parameters)

    # Get data-sets
    data_set_train = dataset_generator([csv_train_file],
                                       parameters,
                                       batch_size=parameters.train_batch_size,
                                       data_augmentation=parameters.data_augmentation,
                                       num_epochs=parameters.n_epochs)

    data_set_eval = dataset_generator([csv_eval_file],
                                      parameters,
                                      batch_size=parameters.eval_batch_size,
                                      data_augmentation=False,
                                      num_epochs=parameters.n_epochs)

    # Train model
    model.fit(data_set_train,
              epochs=epochs,
              initial_epoch=initial_epoch,
              steps_per_epoch=np.floor(n_samples_train / parameters.train_batch_size),
              validation_data=data_set_eval,
              validation_steps=np.floor(n_samples_eval / parameters.eval_batch_size),
              callbacks=list_callbacks)


@click.command()
@click.option('--config_filename', help='A json file containing the config for the experiment')
def configure(config_filename: str):
    ex.add_config(config_filename)
    ex.run(named_configs=[config_filename])


if __name__ == '__main__':
    configure()
