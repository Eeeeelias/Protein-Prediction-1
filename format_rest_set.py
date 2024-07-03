# this is supposed to format the rest set label data (p-scores) so that it can be used interchangably with our
# CNN model. The embeddings do not need changing, only the labels.

import pandas as pd
import ast
import h5py


rest_data = pd.read_csv('data/rest_set_unfiltered.tsv', sep='\t')

rest_data = rest_data[['ID', 'pscores']]

c = 0
for i, row in rest_data.iterrows():
    disorder = row['pscores'].split(",")
    disorder = [float(x) if x != 'NA' else None for x in disorder]
    # check how many values are missing
    missing = sum([1 for x in disorder if x is None])
    # filter out those with more than 30% missing values
    if missing > 0.3 * len(disorder):
        c += 1
        print(f'{row["ID"]} has more than half missing values. Skipping.')
        rest_data.drop(i, inplace=True)
    # backfill the missing values
    tmp_series = pd.Series(disorder)
    disorder = tmp_series.bfill().ffill().tolist()
    rest_data.at[i, 'pscores'] = disorder

print("skipped ", c, " entries")
rest_data.to_csv('data/rest_set_unfiltered_with_dense_format.tsv', sep='\t', index=False)

print(len(rest_data))

h5file = h5py.File('data/trizod_embeddings.h5', 'r')
print(len(h5file.keys()))

print(len(set(rest_data['ID']) & set(h5file.keys())))
