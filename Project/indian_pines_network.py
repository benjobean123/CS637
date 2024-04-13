from torch import nn

class IndianPinesReLUNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



class IndianPinesLeakyNetwork(nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 128),
            nn.LeakyReLU(negative_slope),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope),
            nn.Linear(32, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits