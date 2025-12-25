import torch
from Swiglu import SwigLU

x = torch.rand(2)
s = SwigLU()
out = s(x)
print(out)
y = torch.rand(3)
out2 = s(y)
print(out2)