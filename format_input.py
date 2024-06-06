# format full_filtered.csv file to input format for create_embeddings.py

import pandas as pd
import re

def read_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


def sequence_dict(data):
    return data.set_index('PDBchain')['Sequence'].to_dict()


def create_batches(input_data, batch_size=128):
    num_batches = len(input_data) // batch_size
    batches = []
    for i in range(0, len(input_data), len(input_data) // num_batches):
        batch = {}
        for key in list(input_data.keys())[i:i + len(input_data) // num_batches]:
            batch[key] = input_data[key]
        batches.append(batch)
    return batches