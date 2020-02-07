#!/usr/bin/env python
__author__ = "jeksonic"
__license__ = "GPL"

import os, sys
import re
from glob import glob

import cv2
import click
import pandas as pd
from git import Repo

from alphabet_helpers import generate_alphabet_file
from string_data_manager import tf_crnn_label_formatting


@click.command()
@click.option('--download_dir')
@click.option('--generated_data_dir')
def prepare_open_alpr_data(download_dir: str,
                     generated_data_dir: str):

    # Check if exist
    if os.path.exists(download_dir):
        print('Download directory is not empty.')
        return

    # Download data
    print('Starting downloads...')

    git_url = 'https://github.com/openalpr/benchmarks.git'
    repo_dir = os.path.join(download_dir, 'repo')
    Repo.clone_from(git_url, repo_dir)

    # Generate csv from .txt files
    print('Generating files for the experiment...')

    data_dir = os.path.join(repo_dir, 'endtoend')
    export_img_dir = os.path.join(generated_data_dir, 'generated_img')
    export_csv_dir = os.path.join(generated_data_dir, 'generated_csv')
    os.makedirs(export_img_dir, exist_ok=True)
    os.makedirs(export_csv_dir, exist_ok=True)

    df = pd.DataFrame(columns=['img_url', 'label'])
    for name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, name)
        if os.path.isdir(dir_path) and len(name) == 2:
            for file in glob(os.path.join(dir_path, '*.txt')):
                with open(file, 'r') as text_file:
                    text = text_file.read()
                text = re.split(r'\t|\n| ', text)

                img_file = os.path.join(dir_path, text[0])
                label = text[5]

                x1 = max(int(text[1]), 0)
                x2 = x1 + int(text[3])
                y1 = max(int(text[2]), 0)
                y2 = y1 + int(text[4])

                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                img = img[y1:y2, x1:x2]
                img_file = os.path.join(export_img_dir, text[0])
                cv2.imwrite(img_file, img)

                df = df.append(pd.Series([img_file[1:], label], index=df.columns), ignore_index=True)

    df.sort_values('img_url', inplace=True)

    index = 0
    count = df.shape[0]
    for data_set in [('test', 10), ('validation', 10), ('train', -1)]:
        new_index = count // data_set[1] + index if data_set[1] != -1 else count - 1
        csv_filename = os.path.join(export_csv_dir, '{}.csv'.format(data_set[0]))
        set_df = df.loc[index:new_index, :]
        set_df.to_csv(csv_filename, sep=';', encoding='utf-8', header=False, index=False, escapechar="\\", quoting=3)
        index = new_index + 1

    # Format string label to tf_crnn formatting
    for csv_filename in glob(os.path.join(export_csv_dir, '*')):
        tf_crnn_label_formatting(csv_filename)

    # Generate alphabet
    alphabet_dir = os.path.join(generated_data_dir, 'generated_alphabet')
    os.makedirs(alphabet_dir, exist_ok=True)

    generate_alphabet_file(glob(os.path.join(export_csv_dir, '*')),
                           os.path.join(alphabet_dir, 'open_alpr_alphabet_lookup.json'))


if __name__ == '__main__':
    prepare_open_alpr_data()
