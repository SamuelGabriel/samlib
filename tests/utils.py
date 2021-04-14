import torch

def almost_equal(first: torch.Tensor, second: torch.Tensor, delta=1e-7):
    return ((first-second).abs() <= delta).all()