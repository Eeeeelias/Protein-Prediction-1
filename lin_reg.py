import torch.nn.functional as F
import tqdm

from dataload import Dataloader
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from collections import Counter
import pickle

writer = SummaryWriter('logs/protpred1')


class Lin_reg(nn.Module):  # model class
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = nn.Sequential(
            nn.Linear(self.hparams['input_size'], self.hparams["n_hidden"]),
            nn.Linear(self.hparams['n_hidden'], 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def save_to_file(model, file_name, grid=False):  # function for saving model
    if grid:
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
    else:
        torch.save(model.state_dict(), file_name)


def training(dataset, keys, model, h_params):  # training with cross validation

    batch_size = h_params['batch_size']
    epochs = h_params['epochs']
    best_model = None
    best_val_loss = float('inf')

    losses_train = []
    losses_val = []
    print('starting cross validation')

    train_keys, val_keys = keys[0], keys[1]

    train_loader = Dataloader(dataset, train_keys, batch_size=batch_size, shuffle=True)
    val_loader = Dataloader(dataset, val_keys, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=model.hparams['lr'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        model.train()  # training model
        running_loss = 0.0

        for inputs, targets in tqdm.tqdm(train_loader, maxinterval=len(train_loader)):
            # send data to device
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses_train.append(running_loss / len(train_loader))
        writer.add_scalar('Training loss', running_loss / len(train_loader), epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            running_loss = 0.0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs)
                outputs = outputs.view(-1)
                val_loss += criterion(outputs, targets).item()

                running_loss += loss.item()

        # remember validation scores
        losses_val.append(running_loss / len(val_loader))
        writer.add_scalar('Validation loss', running_loss / len(val_loader), epoch)

        avg_val_loss = val_loss / len(val_loader)

        #early_stopping = EarlyStopping(patience=5, verbose=True)

        '''
        early_stopping(avg_val_loss)
        if early_stopping.should_stop():
            print("Early stopping triggered.")
        break
        '''

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model

    print(f"Best validation loss: {best_val_loss}")
    return best_model, losses_train, losses_val


def predict(dataset, keys, model):
    batch_size = 32

    test_loader = Dataloader(dataset, keys, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions = []
    truths = []
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(test_loader, maxinterval=len(test_loader)):
            inputs = inputs.to(model.device)
            outputs = model(inputs)
            predictions.append(outputs)
            truths.append(targets)

    return predictions, truths


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
