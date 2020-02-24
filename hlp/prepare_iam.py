#!/usr/bin/env python
__author__ = "jeksonic"
__license__ = "GPL"

import os
import click
from glob import glob
from taputapu.databases import iam

from abstract_data_loader import AbstractDataLoader

DATA_NAME = 'iam'


class IamDataLoader(AbstractDataLoader):
    def __init__(self, download_dir: str, generated_data_dir: str):
        super().__init__(DATA_NAME, download_dir, generated_data_dir)

    def download_data(self):
        iam.download(self.download_dir)
        print('Starting extractions...')
        iam.extract(self.download_dir)

    def generate_data(self):
        export_splits_dir = os.path.join(self.generated_data_dir, 'generated_splits')
        os.makedirs(export_splits_dir, exist_ok=True)
        iam.generate_splits_txt(os.path.join(self.download_dir, 'ascii', 'lines.txt'),
                                os.path.join(self.download_dir, 'largeWriterIndependentTextLineRecognitionTask'),
                                export_splits_dir)

        # Generate csv from .txt splits files
        for file in glob(os.path.join(export_splits_dir, '*')):
            export_basename = os.path.basename(file).split('.')[0]
            iam.create_experiment_csv(file,
                                      os.path.join(self.download_dir, 'lines'),
                                      os.path.join(self.export_csv_dir, '{}.csv'.format(export_basename)),
                                      False,
                                      True)


@click.command()
@click.option('--download_dir', default='../data/{}/download'.format(DATA_NAME), show_default=True,
              help='Downloading directory.')
@click.option('--generated_data_dir', default='../data/{}/generated'.format(DATA_NAME), show_default=True,
              help='Directory for generated files.')
def prepare_data(download_dir: str, generated_data_dir: str):
    IamDataLoader(download_dir, generated_data_dir).prepare_structured()


if __name__ == '__main__':
    prepare_data()
