# this is supposed to format the rest set label data (p-scores) so that it can be used interchangably with our
# CNN model. The embeddings do not need changing, only the labels.

import pandas as pd
import ast


rest_data = pd.read_csv('data/trizod_test_set.tsv', sep='\t')

rest_data = rest_data[['ID', 'pscores']]

for i, row in rest_data.iterrows():
    disorder = row['pscores'].split(",")
    disorder = [float(x) if x != 'NA' else None for x in disorder]
    # backfill the missing values
    tmp_series = pd.Series(disorder)
    disorder = tmp_series.bfill().ffill().tolist()
    rest_data.at[i, 'pscores'] = disorder

rest_data.to_csv('data/trizod_test_set_with_dense_format.tsv', sep='\t', index=False)

print(len(rest_data))

import h5py

h5file = h5py.File('data/trizod_embeddings.h5', 'r')
print(len(h5file.keys()))

print(len(set(rest_data['ID']) & set(h5file.keys())))
