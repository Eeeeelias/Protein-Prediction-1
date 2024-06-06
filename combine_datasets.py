# combine datasets of h5 file

import h5py
import numpy as np
import pandas as pd
import ast
import tqdm


h5data = h5py.File('data/embeddings.h5', 'r')

# check if key '1T77A' is in the h5 file
if '1T77A' in h5data:
    print('key exists')
