import pandas as pd
import ast
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from collections import Counter
import pickle


class dense_Dataset:  # dataset class
    def __init__(self, path_to_embed, path_to_dense, test_perc=.15, val_perc=.15):
        self.dense = self.make_dataset(path_to_dense)
        self.embed_file = h5py.File(path_to_embed, 'r')
        # get keys of embed_file
        self.embed_keys = list(self.embed_file.keys())
        self.keys = list(set(self.embed_keys) & set(self.dense.keys()))
        self.train_keys, self.test_keys = train_test_split(self.keys, test_size=test_perc, random_state=42)
        self.train_keys, self.val_keys = train_test_split(self.train_keys, test_size=val_perc, random_state=0)

    @staticmethod
    def make_dataset(path_to_dense):  # creating dataset
        dense = pd.read_csv(path_to_dense, sep='\t')
        # filter for only those with density
        # Apply the conversion to the 'densities' column
        dense = dense[~dense['contact_density'].isna()]
        dense['contact_density'] = dense['contact_density'].apply(ast.literal_eval)
        dense = dense.set_index('PDBchain')['contact_density'].to_dict()
        return dense

    def __len__(self):
        return (len(self.keys))

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.keys[index]
        return torch.tensor(self.embed_file[index], dtype=torch.float), torch.tensor(self.dense[index],
                                                                                     dtype=torch.float)


class Dataloader():
    def __init__(self, dataset, keys, batch_size=32, shuffle=True):
        self.dataset, self.keys, self.batch_size, self.shuffle = dataset, keys, batch_size, shuffle

    def __iter__(self):
        def return_data(batch):
            embeddings = [x[0][0] for x in batch]
            embeddings = torch.vstack(embeddings)
            labels = [x[0][1] for x in batch]
            labels = torch.cat(labels)
            chain_ids = [x[1] for x in batch]
            return embeddings, labels, chain_ids

        batches = []
        if self.shuffle:
            index_iterator = iter(np.random.permutation(self.keys))
        else:
            index_iterator = iter(self.keys)
        batch = []

        for index in index_iterator:
            batch.append((self.dataset[index], index))
            if len(batch) == self.batch_size:
                batches.append(batch)
                yield return_data(batch)
                batch = []
            # if there are any remaining samples
        if len(batch) > 0:
            batches.append(batch)
            yield return_data(batch)

    def __len__(self):
        return (len(self.keys) + self.batch_size - 1) // self.batch_size
