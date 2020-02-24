#!/usr/bin/env python
__author__ = 'jeksonic'
__license__ = "GPL"

import os
import pickle
import pandas as pd
import urllib.request
from glob import glob
from tqdm import tqdm

from alphabet_helpers import generate_alphabet_file
from string_data_manager import tf_crnn_label_formatting
from tf_crnn import CONST


class DownloadProgressBar(tqdm):
    def update_to(self, block_num=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(block_num * block_size - self.n)


class AbstractDataLoader(object):
    """
    Base class for downloading and generating data in suitable format.

    """

    def __init__(self, data_name: str, download_dir: str, generated_data_dir: str):
        self.data_name = data_name
        self.download_dir = download_dir
        self.generated_data_dir = generated_data_dir
        self.export_csv_dir = os.path.join(generated_data_dir, 'generated_csv')

    def prepare_compressed(self):
        # TODO: Implement
        # with open(os.path.join(self.generated_data_dir, data), 'wb') as f:
        #     pickle.dump(data, f)
        pass

    def prepare_structured(self):

        # Download data
        if self.download_dir_exists:
            print('Download directory exists - ignoring downloading.')
        else:
            print('Starting downloading {} data...'.format(self.data_name))
            os.makedirs(self.download_dir, exist_ok=True)
            self.download_data()

        # Generate csv
        if self.generated_data_dir_exists:
            print('Generated data directory exists - ignoring generating.')
        else:
            print('Generating files for the experiment...')
            os.makedirs(self.export_csv_dir, exist_ok=True)
            self.generate_data()

            # Format string label to tf c-rnn formatting
            print('Format string label to tf_crnn formatting...')
            for csv_filename in glob(os.path.join(self.export_csv_dir, '*')):
                tf_crnn_label_formatting(csv_filename)

            # Generate alphabet
            print('Generating alphabet...')
            alphabet_dir = os.path.join(self.generated_data_dir, 'generated_alphabet')
            os.makedirs(alphabet_dir, exist_ok=True)

            generate_alphabet_file(glob(os.path.join(self.export_csv_dir, '*')),
                                   os.path.join(alphabet_dir, 'alphabet_lookup.json'))

    @property
    def download_dir_exists(self):
        return os.path.exists(self.download_dir)

    @property
    def generated_data_dir_exists(self):
        return os.path.exists(self.generated_data_dir)

    def download_data(self):
        pass

    def generate_data(self):
        pass

    @staticmethod
    def download_url(url: str, output_path: str):
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class BaseDataLoader(AbstractDataLoader):
    def __init__(self, data_name: str, download_dir: str, generated_data_dir: str):
        super().__init__(data_name, download_dir, generated_data_dir)
        self.export_img_dir = os.path.join(generated_data_dir, 'generated_img')

    def generate_data(self):
        os.makedirs(self.export_img_dir, exist_ok=True)
        self.prepare_data()

        for (data_set, data) in [('train', self.get_train_data()),
                                 ('validation', self.get_val_data()),
                                 ('test', self.get_test_data())]:
            csv_filename = os.path.join(self.export_csv_dir, '{}.csv'.format(data_set))
            pd.concat([pd.Series(data[0]), pd.Series(data[1])], axis=1) \
                .to_csv(csv_filename, sep=';', encoding=CONST.FILE_ENCODING,
                        header=False, index=False, escapechar="\\", quoting=3)

    def prepare_data(self):
        pass

    def get_train_data(self):
        pass

    def get_val_data(self):
        pass

    def get_test_data(self):
        pass
