import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(300, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 128)
        self.actor = nn.Linear(128, 300)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return self.actor(x)
        
        
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(300, 128)
        self.linear2 = nn.Linear(128, 64)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        return self.critic(x).view(-1)