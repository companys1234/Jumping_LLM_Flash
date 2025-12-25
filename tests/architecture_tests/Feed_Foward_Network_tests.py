import torch
from Feed_Foward_Network import Feed_Forward_Network
from Swiglu import SwigLU

x = torch.rand(128,20)
print(x.shape)
ffn = Feed_Forward_Network(20,30,SwigLU(),True)

out = ffn(x)
print(out)