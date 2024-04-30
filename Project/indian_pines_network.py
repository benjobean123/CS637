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


class IndianPinesLeakySmallNetwork(nn.Module):
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


class IndianPinesLeakyLargeNetwork(nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 180),
            nn.LeakyReLU(negative_slope),
            nn.Linear(180, 150),
            nn.LeakyReLU(negative_slope),
            nn.Linear(150, 120),
            nn.LeakyReLU(negative_slope),
            nn.Linear(120, 90),
            nn.LeakyReLU(negative_slope),
            nn.Linear(90, 60),
            nn.LeakyReLU(negative_slope),
            nn.Linear(60, 30),
            nn.LeakyReLU(negative_slope),
            nn.Linear(30, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class IndianPinesLeakyFinalNetwork(nn.Module):
    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200, 180),
            nn.LeakyReLU(negative_slope),
            nn.Linear(180, 150),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(0.4),
            nn.Linear(150, 120),
            nn.LeakyReLU(negative_slope),
            nn.Linear(120, 90),
            nn.LeakyReLU(negative_slope),
            nn.Linear(90, 60),
            nn.LeakyReLU(negative_slope),
            nn.Linear(60, 30),
            nn.LeakyReLU(negative_slope),
            nn.Linear(30, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits