#!/usr/bin/env python
__author__ = 'solivr'
__license__ = "GPL"

import os
import cv2
import click
import tensorflow as tf
from typing import List
from sklearn.model_selection import train_test_split

from abstract_data_loader import BaseDataLoader

DATA_NAME = 'mnist'


class MnistDataLoader(BaseDataLoader):
    def __init__(self, download_dir: str, generated_data_dir: str):
        super().__init__(DATA_NAME, download_dir, generated_data_dir)
        self.val_ratio = 0.1
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None

    def download_data(self):
        download_path = os.path.abspath(os.path.join(self.download_dir, 'data.npz'))
        data_lib = tf.keras.datasets.mnist
        (x_train, y_train), (self.x_test, self.y_test) = data_lib.load_data(download_path)
        self.x_train, self.x_val, self.y_train, self.y_val \
            = train_test_split(x_train, y_train, test_size=self.val_ratio, random_state=0)

    def get_train_data(self):
        return self._generate_data_set('train', self.x_train, self.y_train)

    def get_val_data(self):
        return self._generate_data_set('validation', self.x_val, self.y_val)

    def get_test_data(self):
        return self._generate_data_set('test', self.x_test, self.y_test)

    def _generate_data_set(self, name: str, x: List[List[List[int]]], y: List[int]):
        print('Generating {} data...'.format(name))
        labels = []
        img_files = []
        for i in range(len(x)):
            label = y[i]
            img_file = os.path.abspath(os.path.join(self.export_img_dir, '{}{}_{}.jpg'.format(name, i + 1, label)))
            cv2.imwrite(img_file, x[i])
            labels.append(label)
            img_files.append(img_file)
        return img_files, labels


@click.command()
@click.option('--download_dir', default='../data/{}/download'.format(DATA_NAME), show_default=True,
              help='Downloading directory.')
@click.option('--generated_data_dir', default='../data/{}/generated'.format(DATA_NAME), show_default=True,
              help='Directory for generated files.')
def prepare_data(download_dir: str, generated_data_dir: str):
    MnistDataLoader(download_dir, generated_data_dir).prepare_structured()


if __name__ == '__main__':
    prepare_data()
