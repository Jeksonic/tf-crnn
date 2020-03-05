#!/usr/bin/env python
__author__ = 'jeksonic'
__license__ = "GPL"

import os
import json
import click
import pandas as pd


@click.command()
@click.option('--result_dir', default='../data/merged', show_default=True, help='Result directory.')
@click.option('--alphabet_data', help='Alphabet files separated with comma.')
@click.option('--train_data', help='Train data files separated with comma.')
@click.option('--test_data', help='Test data files separated with comma.')
@click.option('--val_data', help='Validation data files separated with comma.')
def merge_data(result_dir: str, alphabet_data: str, train_data: str, test_data: str, val_data: str):
    alphabet_files = alphabet_data.split(',')
    train_files = train_data.split(',')
    test_files = test_data.split(',')
    val_files = val_data.split(',')
    if len(set([len(alphabet_files), len(train_files), len(test_files), len(val_files)])) > 1:
        print("Lists has different length.")
        return

    alphabet_data = {}
    train_df = pd.DataFrame(columns=['paths', 'labels'])
    test_df = pd.DataFrame(columns=['paths', 'labels'])
    val_df = pd.DataFrame(columns=['paths', 'labels'])
    for a, tr, t, v in zip(alphabet_files, train_files, test_files, val_files):
        with open(a, 'r', encoding='utf8') as f:
            alphabet = json.load(f)
        for x in alphabet:
            if x not in alphabet_data:
                alphabet_data[x] = len(alphabet_data) + 1
        train_df, test_df, val_df \
            = [df.append(pd.read_csv(csv, sep=';', header=None, names=['paths', 'labels'],
                                     encoding='utf8', escapechar="\\", quoting=0))
               for df, csv in zip([train_df, test_df, val_df], [tr, t, v])]

    os.makedirs(result_dir, exist_ok=True)
    alphabet_file = os.path.join(result_dir, 'alphabet_lookup.json')
    with open(alphabet_file, 'w') as f:
        json.dump(alphabet_data, f)
    for (data_set, data) in [('train', train_df), ('test', test_df), ('validation', val_df)]:
        csv_filename = os.path.join(result_dir, '{}.csv'.format(data_set))
        data.to_csv(csv_filename, sep=';', encoding='utf8', header=False, index=False, escapechar="\\", quoting=3)


if __name__ == '__main__':
    merge_data()
