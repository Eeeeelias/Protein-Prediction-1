# combine datasets of h5 file

import h5py
import numpy as np
import pandas as pd
import ast
import tqdm

rest_data = pd.read_csv('data/trizod_test_set.tsv', sep='\t')

bmrd_to_pdb = rest_data.set_index('ID')['PDBchain'].to_dict()
trizod_keys = set(rest_data['PDBchain'])

key_data = pd.read_csv('data/trizod_test_set_dense_predicted.tsv', sep='\t').dropna()

print(key_data.head())

original_keys = set(key_data['PDBchain'])
key_data = key_data.set_index('PDBchain')['contact_density'].apply(ast.literal_eval)

test_set_keys = set()
# get the keys of the fasta file
with open('udonpred_edit/data/test_set_pp1.fasta') as fasta_file:
    fasta_data = fasta_file.readlines()
    for line in fasta_data:
        if line.startswith('>'):
            test_set_keys.add(line.strip().split()[0][1:])

print(f'Original keys: {len(original_keys)}')
print(f'Trizod keys: {len(trizod_keys)}')
print(f'Intersection: {len(original_keys & trizod_keys)}')

# read the h5 file
with h5py.File('data/trizod_embeddings.h5', 'r') as h5data:
    # create a new h5 file to store the combined data
    with h5py.File('data/trizod_embeddings_test_predicted_1025.h5', 'w') as h5combined:
        for key in tqdm.tqdm(h5data.keys()):
            if bmrd_to_pdb.get(key) and key_data.get(bmrd_to_pdb[key]) and key in test_set_keys:
                new_data = h5data[key][:]
                # add the density data to the new data column wise
                density_data = np.array(key_data[bmrd_to_pdb[key]]).reshape(-1, 1)
                new_data = np.hstack([new_data, density_data])
                h5combined.create_dataset(key, data=new_data)
