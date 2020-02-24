#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
from glob import glob

import click
from tf_crnn.callbacks import CustomPredictionSaverCallback
from tf_crnn.config import Params, CONST
from tf_crnn.data_handler import dataset_generator
from tf_crnn.model import get_model_inference


@click.command()
@click.option('--csv_filename', help='A csv file containing the path to the images to predict')
@click.option('--output_model_dir', help='Directory where all the exported data related to an experiment has been saved')
def prediction(csv_filename: str, output_model_dir: str):
    parameters = Params.from_json_file(os.path.join(output_model_dir, CONST.CONFIG_FILENAME))

    saving_dir = os.path.join(output_model_dir, CONST.FOLDER_SAVED_MODEL)
    last_time_stamp = str(max([int(p.split(os.path.sep)[-1].split('-')[0])
                          for p in glob(os.path.join(saving_dir, '*'))]))
    model = get_model_inference(parameters, os.path.join(saving_dir, last_time_stamp, CONST.MODEL_WEIGHTS_FILENAME))

    data_set_test = dataset_generator([csv_filename],
                                      parameters,
                                      use_labels=False,
                                      batch_size=parameters.eval_batch_size,
                                      shuffle=False)

    ps_callback = CustomPredictionSaverCallback(output_model_dir, parameters)

    _, _, _ = model.predict(x=data_set_test, callbacks=[ps_callback])


if __name__ == '__main__':
    prediction()
