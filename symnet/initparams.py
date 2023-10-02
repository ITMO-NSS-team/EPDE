import torch


def initexpr(model):
    for p in model.parameters():
        p.data = torch.randn(*p.shape,dtype=p.dtype,device=p.device)*1e-1