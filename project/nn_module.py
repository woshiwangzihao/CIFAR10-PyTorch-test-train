from torch import nn
import torch

class Tudui(nn.Module):
    def __init__(self) :
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
