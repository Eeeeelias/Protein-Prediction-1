import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import h5py


class Random_Model():

    def __init__(self):
        dt = pd.read_csv('Protein-Prediction-1/data/densities.csv', sep='\t')
        dt = dt[~dt['contact_density'].isna()]
        dt['contact_density'] = dt['contact_density'].apply(ast.literal_eval)
        self.dens_list = self.extract_densities_per_index(dt['contact_density'])
    @staticmethod
    def extract_densities_per_index(df_column):
        distrib = []
        for i in range(999):
            distrib.append(np.array([lst[i] for lst in df_column if len(lst) > i]))
        return distrib

    def predict(self, input):
        output = []
        for i in range(input.shape[0]):
            output.append(np.random.choice(self.dens_list[i]))
            
        return torch.tensor(np.array(output))

class CNN_Dataloader():
    def __init__(self, dataset, keys, batch_size=32, shuffle=True, window_size = 9):
        self.dataset, self.keys, self.batch_size, self.shuffle = dataset, keys, batch_size, shuffle
        self.window_size = window_size

    def pad_batch(self, batch):
        embeddings, densities, keys = batch
        padded_embeddings, padded_densities = [], []
        max_len = max([embed.shape[0] for embed in embeddings])
        for embed, dense in zip(embeddings, densities):
            pad_embed = F.pad(embed, (0, 0, 0, max_len - embed.shape[0]), 'constant', 0)
            padded_embeddings.append(pad_embed)
            pad_dense = F.pad(dense, (0, max_len - dense.shape[0]), 'constant', 0)
            padded_densities.append(pad_dense)
        return torch.stack(padded_embeddings), torch.stack(padded_densities), keys

    def __iter__(self):
        def return_data(batch):
            embeddings = [x[0][0] for x in batch]
            # embeddings = torch.vstack(embeddings)
            labels = [x[0][1] for x in batch]
            # labels = torch.cat(labels)
            chain_ids = [x[1] for x in batch]
            return embeddings, labels, chain_ids

        batches = []
        if self.shuffle:
            index_iterator = iter(np.random.permutation(self.keys))
        else:
            index_iterator = iter(self.keys)
        batch = []

        for index in index_iterator:
            batch.append((self.dataset[index], index))
            if len(batch) == self.batch_size:
                batches.append(batch)
                yield self.pad_batch(return_data(batch))
                batch = []
            # if there are any remaining samples
        if len(batch) > 0:
            batches.append(batch)
            yield self.pad_batch(return_data(batch))
    def __len__(self):
        return (len(self.keys) + self.batch_size - 1) // self.batch_size


class CNN(nn.Module):
    def __init__(self, hparams):
        super(CNN, self).__init__()
        self.hparams = hparams
        self.device = self.hparams['device']
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 1024, L)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, L, 256)
        x = self.fc1(x)
        return x.squeeze(2)


def train_CNN(train_loader, val_loader, hparams):
    # actual training
    model = CNN(hparams)

    path = "logs/protpred1_cnn"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')

    tb_logger = SummaryWriter(path)

    epochs = model.hparams['epochs']
    best_val_loss = float('inf')

    losses_train = []
    losses_val = []
    print('starting cross validation')

    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()  # training model
        running_loss = 0.0

        for inputs, targets, _ in train_loader:
            # send data to device
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            outputs = outputs.view(-1)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(running_loss)
        losses_train.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_running_loss = 0.0
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(model.device), targets.to(model.device)
                outputs = model(inputs)
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                val_loss += criterion(outputs, targets).item()

                val_running_loss += val_loss

        # remember validation scores
        losses_val.append(val_running_loss / len(val_loader))
        tb_logger.add_scalar('Validation loss', val_running_loss / len(val_loader), epoch)

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model

    print(f"Best validation loss: {best_val_loss}")
