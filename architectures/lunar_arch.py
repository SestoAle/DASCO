import torch.nn as nn
import torch.nn.functional as F

class PolicyEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(PolicyEmbedding, self).__init__()
        self.state_dim = state_dim

        self.emb_1 = nn.Linear(self.state_dim, 256)
        self.emb_2 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        emb = F.relu(self.emb_1(state))
        emb = F.relu(self.emb_2(emb))
        return emb

class CriticEmbedding(nn.Module):
    def __init__(self, state_dim, **kwargs):
        super(CriticEmbedding, self).__init__()
        self.state_dim = state_dim

        self.emb_1 = nn.Linear(self.state_dim, 256)
        self.emb_2 = nn.Linear(256, 256)

        self.output_dim = 256

    def forward(self, state):
        emb = F.relu(self.emb_1(state))
        emb = F.relu(self.emb_2(emb))
        return emb