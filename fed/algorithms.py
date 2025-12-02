# fed/algorithms.py
from copy import deepcopy
from typing import Dict, Any
import torch
import torch.nn as nn

def local_update_fedavg(model: nn.Module,
                        dataloader,
                        device: str,
                        epochs: int = 1,
                        lr: float = 0.01) -> nn.Module:
    """
    Run local training on client's data using standard ERM (FedAvg).
    Returns the updated model.
    """
    model = deepcopy(model)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return model

def local_update_fedprox(model: nn.Module,
                         global_model: nn.Module,
                         dataloader,
                         device: str,
                         epochs: int = 1,
                         lr: float = 0.01,
                         mu: float = 0.001) -> nn.Module:
    """
    Local update for FedProx: adds proximal term mu/2 * ||w - w_global||^2
    """
    model = deepcopy(model)
    model.to(device)
    global_model = deepcopy(global_model).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    global_params = dict(global_model.named_parameters())

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)

            # proximal term
            prox = 0.0
            for (name, param) in model.named_parameters():
                prox += ((param - global_params[name]) ** 2).sum()
            loss = loss + (mu / 2.0) * prox

            loss.backward()
            optimizer.step()

    return model

def aggregate_models(global_model: nn.Module,
                     client_models: Dict[int, nn.Module],
                     client_weights: Dict[int, int]) -> nn.Module:
    """
    Standard FedAvg aggregation:
    global = sum_i ( (n_i / sum_j n_j) * local_i )

    Only averages floating-point tensors. Non-floating tensors (e.g.
    BatchNorm's num_batches_tracked) are just copied from one client.
    """
    global_state = global_model.state_dict()
    total_weight = sum(client_weights.values())

    # Initialize aggregated state
    agg_state = {}
    for k, v in global_state.items():
        if torch.is_floating_point(v):
            agg_state[k] = torch.zeros_like(v)
        else:
            # just clone the original value; we'll overwrite with a client's
            agg_state[k] = v.clone()

    # Weighted sum over clients
    for cid, local_model in client_models.items():
        local_state = local_model.state_dict()
        w = client_weights[cid] / total_weight

        for k in agg_state.keys():
            if torch.is_floating_point(agg_state[k]):
                # weighted average for float tensors
                agg_state[k] += w * local_state[k].to(agg_state[k].dtype)
            else:
                # for int / bool / etc. just take one client's value
                # (they should all be very similar / identical)
                agg_state[k] = local_state[k]

    global_model.load_state_dict(agg_state)
    return global_model
