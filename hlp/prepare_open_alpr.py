#!/usr/bin/env python
__author__ = "jeksonic"
__license__ = "GPL"

import os
import re
import cv2
import click
from git import Repo
from glob import glob

from abstract_data_loader import BaseDataLoader

DATA_NAME = 'open_alpr'


class OpenAlprDataLoader(BaseDataLoader):
    def __init__(self, download_dir: str, generated_data_dir: str):
        super().__init__(DATA_NAME, download_dir, generated_data_dir)
        self.repo_dir = os.path.join(download_dir, 'repo')
        self.data = []
        self.data_ratios = {'test': 10, 'validation': 10, 'train': -1}

    def download_data(self):
        git_url = 'https://github.com/openalpr/benchmarks.git'
        Repo.clone_from(git_url, self.repo_dir)

    def prepare_data(self):
        data_dir = os.path.join(self.repo_dir, 'endtoend')
        for name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, name)
            if os.path.isdir(dir_path) and len(name) == 2:
                for file in glob(os.path.join(dir_path, '*.txt')):
                    with open(file, 'r') as text_file:
                        text = text_file.read()
                    text = re.split(r'[\t\n ]', text)

                    img_file = os.path.join(dir_path, text[0])
                    label = text[5]

                    x1 = max(int(text[1]), 0)
                    x2 = x1 + int(text[3])
                    y1 = max(int(text[2]), 0)
                    y2 = y1 + int(text[4])

                    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                    img = img[y1:y2, x1:x2]
                    img_file = os.path.abspath(os.path.join(self.export_img_dir, text[0]))
                    cv2.imwrite(img_file, img)

                    self.data.append((img_file, label))
        self.data.sort(key=lambda tup: tup[0])

    def get_train_data(self):
        return self._generate_data_set('train')

    def get_val_data(self):
        return self._generate_data_set('validation')

    def get_test_data(self):
        return self._generate_data_set('test')

    def _generate_data_set(self, name: str):
        print('Generating {} data...'.format(name))

        index = 0
        count = len(self.data)
        for key, ratio in self.data_ratios.items():
            new_index = count // ratio + index if ratio != -1 else count
            if key == name:
                data = self.data[index:new_index]
                x, y = map(list, zip(*data))
                return x, y
            index = new_index


@click.command()
@click.option('--download_dir', default='../data/{}/download'.format(DATA_NAME), show_default=True,
              help='Downloading directory.')
@click.option('--generated_data_dir', default='../data/{}/generated'.format(DATA_NAME), show_default=True,
              help='Directory for generated files.')
def prepare_data(download_dir: str, generated_data_dir: str):
    OpenAlprDataLoader(download_dir, generated_data_dir).prepare_structured()


if __name__ == '__main__':
    prepare_data()
