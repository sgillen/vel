import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import softmax
import torch


from vel.api import LinearBackboneModel

class Router(LinearBackboneModel):
    def __init__(self, input_size:int, output_size:int , router_size:int, hidden_size:int):
        # Routing layer gates
        self.r_linear1 = nn.Linear(input_size, router_size)
        self.r_linear2 = nn.Linear(router_size, 2)

        # Swingup layer gates
        self.s_linear1 = nn.Linear(input_size, hidden_size)
        self.s_linear2 = nn.Linear(hidden_size, output_size)

        # This is basically our static gain matrix (maybe I should make this a matrix rather than a linear layer...)
        self.k = nn.Linear(input_size, output_size, bias=False)

        # Required for the training
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # Gating
        g = torch.sigmoid(self.r_linear1(x))
        g = torch.sigmoid(self.r_linear2(g))
        d = softmax(g, dim=-1)

        # Swingup
        s = torch.relu(self.s_linear1(x))
        ys = self.s_linear2(s)

        # Balance
        yb = self.k(x)

        return ys, yb, d



    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        return self.hidden_units

    super().__init__()

