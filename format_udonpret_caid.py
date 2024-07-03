import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, f1_score


outputs = {}
with open('data/udonpred_original_all.caid', 'r') as f:
    current_chain = []
    current_id = ''
    for line in f.readlines():
        if line.startswith('>'):
            if current_chain:
                outputs[current_id] = current_chain
            current_id = line.strip()[1:]
            current_chain = []
        else:
            current_chain.append(float(line.strip().split("\t")[2]))
    outputs[current_id] = current_chain

pscores = {}
ground_truth = pd.read_csv('data/trizod_test_set.tsv', sep='\t').dropna()
for i, row in ground_truth.iterrows():
    row_scores = row['pscores'].split(",")
    for x in range(len(row_scores)):
        row_scores[x] = float(row_scores[x]) if row_scores[x] != 'NA' else 'NA'
    pscores[row['ID']] = row_scores


# go through p scores, if value is na, remove from both pscores and outputs
na_removed_pscores = {}
na_removed_outputs = {}
for key, value in pscores.items():
    new_value = []
    new_output = []
    for i in range(len(value)):
        if value[i] == "NA":
            continue
        new_value.append(value[i])
        new_output.append(outputs[key][i])
    na_removed_pscores[key] = new_value
    na_removed_outputs[key] = new_output

# flatten the outputs and pscores and calculate mse score
flat_outputs = []
flat_pscores = []
for key, value in na_removed_pscores.items():
    flat_outputs.extend(na_removed_outputs[key])
    flat_pscores.extend(value)

mse = sum([(x - y) ** 2 for x, y in zip(flat_outputs, flat_pscores)]) / len(flat_outputs)
print(mse)


truths = flat_pscores[:200]
preds = flat_outputs[:200]
mins = [min(truths[i], preds[i]) for i in range(len(truths))]
maxs = [max(truths[i], preds[i]) for i in range(len(truths))]
line_color = ['purple' if preds[i] < truths[i] else 'yellow' for i in range(len(truths))]

plt.figure(figsize=(11, 5))
plt.scatter(range(len(preds)), preds, c='purple', label="Predicted Disorder")
plt.scatter(range(len(truths)), truths, c="yellow", label="True Disorder")
plt.vlines(x=range(len(preds)), linestyle='-', linewidth=1, ymin=mins, ymax=maxs, colors=line_color)
plt.xlabel("Residue Index")
plt.ylabel("Disorder")
plt.title("Disorder per Residue")
plt.legend()
plt.show()

truths = np.array(flat_pscores)
preds = np.array(flat_outputs)

true_classes = (truths > 0.5).astype(int)
predicted_classes = (preds > 0.5).astype(int)

auc = roc_auc_score(true_classes, preds)
f1 = f1_score(true_classes, predicted_classes)
print(f'AUC: {auc}')
print(f'F1: {f1}')
