#!/usr/bin/env python
__author__ = "jeksonic"
__license__ = "GPL"

import os
import cv2
import click
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

from abstract_data_loader import BaseDataLoader

DATA_NAME = 'iwt'


class IwtDataLoader(BaseDataLoader):
    def __init__(self, download_dir: str, generated_data_dir: str):
        super().__init__(DATA_NAME, download_dir, generated_data_dir)
        self.data_ratios = {'test': 0.1, 'validation': 0.1, 'train': 0.8}
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None

    def download_data(self):
        file_path = os.path.join(self.download_dir, '2017-IWT4S-CarsReId_LP-dataset.zip')
        self.download_url('https://medusa.fit.vutbr.cz/traffic/download/512/', file_path)
        print('Starting extractions...')
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.download_dir)

    def prepare_data(self):
        x, y = [], []
        csv_file = os.path.join(self.download_dir, 'trainVal.csv')
        df = pd.read_csv(csv_file)
        for i, row in df.iterrows():
            if i % 1000 == 0:
                print("Processed {} rows.".format(i))

            src_file = os.path.join(self.download_dir, row['image_path'])
            src_filename = os.path.basename(src_file).split('.')[0]
            dst_filename = '{}_{}.jpg'.format(row['lp'], src_filename)
            dst_file = os.path.abspath(os.path.join(self.export_img_dir, dst_filename))

            img = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(dst_file, img)

            x.append(dst_file)
            y.append(row['lp'])

        non_train_ratio = self.data_ratios['test'] + self.data_ratios['validation']
        val_ratio = self.data_ratios['validation'] / non_train_ratio
        self.x_train, x_part, self.y_train, y_part = train_test_split(x, y, test_size=non_train_ratio, random_state=0)
        self.x_test, self.x_val, self.y_test, self.y_val \
            = train_test_split(x_part, y_part, test_size=val_ratio, random_state=0)

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test


@click.command()
@click.option('--download_dir', default='../data/{}/download'.format(DATA_NAME), show_default=True,
              help='Downloading directory.')
@click.option('--generated_data_dir', default='../data/{}/generated'.format(DATA_NAME), show_default=True,
              help='Directory for generated files.')
def prepare_data(download_dir: str, generated_data_dir: str):
    IwtDataLoader(download_dir, generated_data_dir).prepare_structured()


if __name__ == '__main__':
    prepare_data()
