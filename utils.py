import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
from torch.utils.tensorboard import SummaryWriter


# custom early stopping, based on chosen metric, works for minimizing metrics
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


class Random_Model():

    def __init__(self):
        dt = pd.read_csv('data/densities.csv', sep='\t')
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
        for i in range(len(input)):
            output.append(np.random.choice(self.dens_list[i]))

        return torch.tensor(np.array(output))


class CNN_Dataloader():
    def __init__(self, dataset, keys, batch_size=32, shuffle=True, window_size=9):
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


class CNNBatchNorm(nn.Module):
    def __init__(self, hparams):
        super(CNNBatchNorm, self).__init__()
        self.hparams = hparams
        self.device = self.hparams['device']
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(256, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, 1024, L)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, L, 256)
        x = self.fc1(x)
        return x.squeeze(2)


def train_CNN(train_loader, val_loader, hparams):
    # actual training
    model = CNN(hparams)
    model.to(model.device)

    path = "logs/protpred1_cnn"
    num_of_runs = len(os.listdir(path)) if os.path.exists(path) else 0
    path = os.path.join(path, f'run_{num_of_runs + 1}')

    tb_logger = SummaryWriter(path)
    early_stop = EarlyStopping(patience=hparams['patience'])

    epochs = model.hparams['epochs']
    best_val_loss = float('inf')

    losses_train = []
    losses_val = []
    print('starting training')

    optimizer = torch.optim.Adam(model.parameters(), lr=model.hparams['lr'])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", verbose=True, factor=0.5, min_lr=1e-5,
                                                           patience=5)

    for epoch in range(epochs):
        model.train()  # training model
        running_loss = 0.0

        for inputs, targets, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}",
                                            maxinterval=len(train_loader)):
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
        losses_train.append(running_loss / len(train_loader))
        tb_logger.add_scalar('Training loss', running_loss / len(train_loader), epoch)

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
        avg_val_loss = val_loss / len(val_loader)
        losses_val.append(avg_val_loss)
        tb_logger.add_scalar('Validation loss', avg_val_loss, epoch)

        early_stop(avg_val_loss)
        scheduler.step(metrics=avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    print(f"Best validation loss: {best_val_loss}")
    return model, losses_train, losses_val, outputs, targets


def load_model(model_path, hparams):
    model = CNN(hparams)
    model.load_state_dict(torch.load(model_path))
    model.to(model.device)
    model.eval()
    return model
