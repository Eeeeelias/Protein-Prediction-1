# combine datasets of h5 file

import h5py
import numpy as np
import pandas as pd
import ast
import tqdm

key_data = pd.read_csv('data/rest_set.tsv', sep='\t')

trizod_keys = set(key_data['PDBchain'])

key_data = pd.read_csv('data/densities.csv', sep='\t')

original_keys = set(key_data['PDBchain'])

print(f'Original keys: {len(original_keys)}')
print(f'Trizod keys: {len(trizod_keys)}')
print(f'Intersection: {len(original_keys & trizod_keys)}')

# # read the h5 file
# with h5py.File('data/trizod_embeddings.h5', 'r') as h5data:
#     # create a new h5 file to store the combined data
#     with h5py.File('data/trizod_embeddings_pdb.h5', 'w') as h5combined:
#         for key in tqdm.tqdm(h5data.keys()):
#             if key in key_data:
#                 h5combined.create_dataset(key_data[key], data=h5data[key][:])
#             else:
#                 print(f'{key} not found in key data')
#