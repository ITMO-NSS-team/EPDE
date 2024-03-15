import torch


def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-3
    for p in list(model.parameters()):
        p = p.abs()
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss


def loss(model, u_left, u_right, block, sparsity):
    stepnum = block if block >= 1 else 1

    dataloss = 0
    sparseloss = _sparse_loss(model)

    u_der = u_left
    for steps in range(1, stepnum + 1):
        u_dertmp = model(u_right)

        dataloss = dataloss + \
                   torch.mean((u_dertmp - u_der) ** 2)
        # layerweight[steps-1]*torch.mean(((uttmp-u_obs[steps])/(steps*dt))**2)
        # ut = u_right
    loss = dataloss + stepnum * sparsity * sparseloss
    return loss