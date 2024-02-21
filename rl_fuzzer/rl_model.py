import torch.nn as nn
import torch.functional as F

class ActorCritic(nn.Module):
    def __init__(self):
        self.linear1 = nn.Linear(300, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 128)

        self.actor = nn.Linear(128, 300)
        self.critic = nn.Linear(128, 1)
        self.predictor = nn.Linear(128, 300)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.relu(x)

        x = self.linear3(x)
        x = F.relu(x)

        return x
    
    def get_action(self, x):
        x = self(x)
        