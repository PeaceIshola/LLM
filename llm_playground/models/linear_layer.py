import torch
import torch.nn as nn

# Define a MyLinear PyTorch module and perform y = Wx + b.

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        # Initialize weights and bias as learnable parameters.
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Matrix multiplication followed by bias addition
        return torch.matmul(self.weight, x) + self.bias    # y = Wx + b


lin = MyLinear(3, 2)
x = torch.tensor([1.0, -1.0, 0.5])
print("Input :", x)
print("Weights:", lin.weight)
print("Bias   :", lin.bias)
print("Output :", lin(x))


# Create a linear layer using pytorch's nn.Linear
lin = nn.Linear(in_features=3, out_features=2)

x = torch.tensor([1.0, -1.0, 0.5])
print("Input :", x)
print("Weights:", lin.weight)
print("Bias   :", lin.bias)
print("Output :", lin(x))
