import torch
import torch.nn as nn


class NeuralNet(nn.Module):  # model class
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.device = hparams.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model = nn.Sequential(
            nn.Linear(self.hparams['input_size'], self.hparams["n_hidden"]),
            nn.ReLU(),
            nn.Linear(self.hparams['n_hidden'], 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x
