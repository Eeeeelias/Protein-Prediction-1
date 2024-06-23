# combine datasets of h5 file

import h5py
import numpy as np
import pandas as pd
import ast
import tqdm

rest_data = pd.read_csv('data/rest_set.tsv', sep='\t')

bmrd_to_pdb = rest_data.set_index('ID')['PDBchain'].to_dict()
trizod_keys = set(rest_data['PDBchain'])

key_data = pd.read_csv('data/rest_set_dense.tsv', sep='\t').dropna()

original_keys = set(key_data['PDBchain'])
key_data = key_data.set_index('PDBchain')['contact_density'].apply(ast.literal_eval)

print(f'Original keys: {len(original_keys)}')
print(f'Trizod keys: {len(trizod_keys)}')
print(f'Intersection: {len(original_keys & trizod_keys)}')

# read the h5 file
with h5py.File('data/trizod_embeddings.h5', 'r') as h5data:
    # create a new h5 file to store the combined data
    with h5py.File('data/trizod_embeddings_pdb.h5', 'w') as h5combined:
        for key in tqdm.tqdm(h5data.keys()):
            if bmrd_to_pdb.get(key) and key_data.get(bmrd_to_pdb[key]):
                new_data = h5data[key][:]
                # add the density data to the new data column wise
                density_data = np.array(key_data[bmrd_to_pdb[key]]).reshape(-1, 1)
                new_data = np.hstack([new_data, density_data])
                h5combined.create_dataset(bmrd_to_pdb[key], data=new_data)
            else:
                print(f'{key} not found in key data')

