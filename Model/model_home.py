import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1792), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
                    #nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                    nn.Linear(64, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out

class MLP2(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP2, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1024), nn.PReLU(), nn.Dropout(),
                    #nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
                    #nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                    nn.Linear(64, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out

class MLP3(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP3, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1024), nn.PReLU(), nn.Dropout(),
                    #nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
                    #nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                    nn.Linear(64, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out


class MLP4(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP4, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1792), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1792, 1024), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 768), nn.PReLU(), nn.Dropout(),
                    nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(),
                    nn.Linear(64, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out

class MLP5(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP5, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, 2560), nn.PReLU(), nn.Dropout(),
                    nn.Linear(2560, 1024), nn.PReLU(), nn.Dropout(),
                    nn.Linear(1024, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 512), nn.PReLU(), nn.Dropout(),
                    nn.Linear(512, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
                    nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 128), nn.PReLU(), nn.Dropout(),
                    nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
                    nn.Linear(64, 64), nn.PReLU(), nn.Dropout(),
                    nn.Linear(64, 32), nn.PReLU(), nn.Dropout(),
                    nn.Linear(32, 32), nn.PReLU(),
                    nn.Linear(32, output_size)
                )

    def forward(self, x):
        out = self.fc(x)
        return out
