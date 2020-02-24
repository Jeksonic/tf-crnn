#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from typing import List
import csv
import json
import numpy as np
import pandas as pd


def get_alphabet_units_from_input_data(csv_filename: str,
                                       split_char: str='|'):
    """
    Get alphabet units from the input_data csv file (which contains in each row the tuple
    (filename image segment, transcription formatted))

    :param csv_filename: csv file containing the input data
    :param split_char: splitting character in input_data separting the alphabet units
    :return:
    """
    df = pd.read_csv(csv_filename, sep=';', header=None, names=['image', 'labels'],
                     encoding='utf8', escapechar="\\", quoting=3)
    transcriptions = list(df.labels.apply(lambda x: x.split(split_char)))

    unique_units = np.unique([chars for list_chars in transcriptions for chars in list_chars])

    return unique_units


def generate_alphabet_file(csv_filenames: List[str],
                           alphabet_filename: str):
    """

    :param csv_filenames:
    :param alphabet_filename:
    :return:
    """
    symbols = list()
    for file in csv_filenames:
        symbols.append(get_alphabet_units_from_input_data(file))

    alphabet_units = np.unique(np.concatenate(symbols))

    alphabet_lookup = dict([(au, i+1)for i, au in enumerate(alphabet_units)])

    with open(alphabet_filename, 'w') as f:
        json.dump(alphabet_lookup, f)


def get_abbreviations_from_csv(csv_filename: str) -> List[str]:
    with open(csv_filename, 'r', encoding='utf8') as f:
        csvreader = csv.reader(f, delimiter='\n')
        alphabet_units = [row[0] for row in csvreader]
    return alphabet_units
