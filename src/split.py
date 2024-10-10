# split the raw data
# save in data/processed

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def split_save(config_path):
    config = read_params(config_path)
    test_data_path = config['split_data']['test_path']
    train_data_path = config['split_data']['train_path']
    raw_data_path = config['load_data']['raw_dataset_csv']
    split_ratio = config['split_data']['test_size']
    random_state =  config['base']['random_state']