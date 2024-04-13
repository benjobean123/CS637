from torch import nn

class IndianPinesNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits