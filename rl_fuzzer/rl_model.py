import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        )
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 128)
        self.actor = nn.Linear(128, 300)
        if weights:
            self.load_state_dict(torch.load(weights))

    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.linear1(x.mean(axis=1))
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return self.actor(x).view(-1)
        
        
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        )
        self.linear1 = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x.unsqueeze(0))
        x = self.linear1(x.mean(axis=1))
        x = F.relu(x)
        return self.critic(x).view(-1)