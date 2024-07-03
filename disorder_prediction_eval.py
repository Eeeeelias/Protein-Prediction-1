# read in the pkl file with predictions
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, f1_score
import matplotlib.pyplot as plt

predictions = pickle.load(open('trizod_test_set_disorder_cpu_11.pkl', 'rb'))

# read original disorder data

original_disorder = pd.read_csv('data/trizod_test_set.tsv', sep='\t')

for i, row in original_disorder.iterrows():
    disorder = row['pscores'].split(",")
    disorder = [float(x) if x != 'NA' else None for x in disorder]
    original_disorder.at[i, 'pscores'] = disorder

# flatten the predictions

flat_predictions = []
flat_ids = []
for batch in predictions['test_preds']:
    for pred in batch:
        # predictions were sqrted, so square them
        # pred = [x ** 2 for x in pred]
        flat_predictions.append(pred)
for key in predictions['test_ids']:
    flat_ids.extend(key)


print(f'Length of flat predictions: {len(flat_predictions)}')
print(f'Length of flat ids: {len(flat_ids)}')

# get id from original disorder and cut off the padding
len_dict = original_disorder.set_index('ID')['len'].to_dict()

flat_predictions = [pred[:len_dict[chain]] for pred, chain in zip(flat_predictions, flat_ids)]

# go through flat predictions, the ids and the original disorder and calculate mse over all residues while ignoring
# predictions where the original value is None
raw_predictions = []
raw_truths = []
for pred, chain, orig in zip(flat_predictions, flat_ids, original_disorder['pscores']):
    for p, o in zip(pred, orig):
        if o is not None:
            raw_predictions.append(p)
            raw_truths.append(o)

# calculate mse

mse = mean_squared_error(raw_truths, raw_predictions)
print(f'MSE: {mse}')
r2 = r2_score(raw_truths, raw_predictions)
print(f'R2: {r2}')


raw_truths = np.array(raw_truths)

truths = raw_truths[:88]
preds = raw_predictions[:88]
mins = [min(truths[i], preds[i]) for i in range(len(truths))]
maxs = [max(truths[i], preds[i]) for i in range(len(truths))]
# choose line color to be blue if the prediction is less than the truth, else choose red
line_color = ['#5783A2' if preds[i] < truths[i] else 'tomato' for i in range(len(truths))]

plt.figure(figsize=(11, 5))
plt.scatter(range(len(preds)), preds, label="Predicted Disorder")
plt.scatter(range(len(truths)), truths, c="tomato", label="True Disorder")
plt.vlines(x=range(len(preds)), linestyle='-', linewidth=1, ymin=mins, ymax=maxs, colors=line_color)
plt.xlabel("Residue Index")
plt.ylabel("Contact Density")
plt.title("Disorder per Residue")
plt.legend()
plt.show()

# do histogram of raw truths

# split into two parts: >0.5 in truth is disorder, <0.5 is order, calculate F1 score
truths = np.array(raw_truths)
preds = np.array(raw_predictions)

true_classes = (truths > 0.5).astype(int)
predicted_classes = (preds > 0.5).astype(int)

auc = roc_auc_score(true_classes, preds)
f1 = f1_score(true_classes, predicted_classes)
print(f'AUC: {auc}')
print(f'F1: {f1}')
print(f"Max: {max(raw_predictions)}, Max true: {max(raw_truths)}")