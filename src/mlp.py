import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, args, nfeat):
        super(MLP, self).__init__()
        self.nhid1 = args.hidden1
        self.out_dim = 1
        self.layer1 = nn.Linear(nfeat, self.nhid1)
        self.layer2 = nn.Linear(self.nhid1, self.out_dim)
    def forward(self, x):
        x = self.layer1(x)
        output_emb = self.layer2(x)
        return x, output_emb
