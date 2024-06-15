import numpy as np
import scipy
import torch
import torch.nn as nn
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import r2_score


class Lin_reg(nn.Module):  # model class
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = nn.Sequential(
            nn.Linear(self.hparams['input_size'], self.hparams["n_hidden"]),
            nn.Linear(self.hparams['n_hidden'], self.hparams["n_hidden"]),
            nn.Linear(self.hparams['n_hidden'], self.hparams["n_hidden"]),
            nn.Linear(self.hparams['n_hidden'], 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def evaluate(true, predicted):
    # flatten the lists
    true = torch.cat(true)
    true = true.view(-1)
    true = true.to('cpu')
    # convert to numpy
    true = true.numpy()
    predicted = torch.cat(predicted)
    predicted = predicted.numpy()
    pearson = scipy.stats.pearsonr(true, predicted)
    r2_scor = r2_score(true, predicted)
    mse = np.mean((true - predicted) ** 2)
    return r2_scor, pearson, mse


# needs to be adjusted for other scores
def bootstrap(true, predicted, n):  # function for getting bootstrapped scores to produce confidence intervals
    accs = []
    mccs = []
    for i in range(n):
        # Randomly sample with replacement
        indices = np.random.choice(len(true), size=len(true), replace=True)

        # Create bootstrap samples
        true_bootstrap = [true[i] for i in indices]
        predicted_bootstrap = [predicted[i] for i in indices]

        # Calculate metric for bootstrap sample
        accs.append(accuracy_score(true_bootstrap, predicted_bootstrap) * 100)
        mccs.append(matthews_corrcoef(true_bootstrap, predicted_bootstrap) * 100)
        #f1s.append(f1_score(true_labels_bootstrap, predicted_labels_bootstrap, average='micro')*100)
    # Analyze the distribution of the metric (e.g., calculate confidence intervals)
    return mccs, accs


def testing(test_X, model, grid=False):
    test_X = torch.as_tensor(test_X, dtype=torch.float32)
    if (grid):  # grid model from skorch uses different function to make predictions
        outputs = torch.as_tensor(model.predict_proba(test_X))
        _, predicted = torch.max(outputs, dim=1)
    else:
        model.eval()  # set the model to evaluation mode
        outputs = model(test_X)
        # check if dimensions is greater than 1
        if len(outputs.shape) > 1:
            _, predicted = torch.max(outputs, dim=1)
        else:
            predicted = outputs
    #print(predicted)
    return predicted


class EarlyStopping:  # Early stopping to break out of epoch loop when loss doesn't improve over x number of epochs (patience)
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

    def should_stop(self):
        return self.early_stop
