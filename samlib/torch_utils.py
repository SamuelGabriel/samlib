import torch


def get_gradients(model, copy=False, as_vector=False):
    grad_list = []
    for param in model.parameters():
        g = param.grad
        if copy and g is not None:
            g = g.clone()
        grad_list.append(g)
    if as_vector:
        return torch.cat([g.flatten() for g in grad_list if g is not None])
    return grad_list

def to_tensor_if_needed(tensor_or_list):
    if isinstance(tensor_or_list,torch.Tensor):
        return tensor_or_list
    else:
        return torch.tensor(tensor_or_list)

def denormalize(img, mean, std):
    mean, std = to_tensor_if_needed(mean).to(img.device), to_tensor_if_needed(std).to(img.device)
    return img.mul_(std[:,None,None]).add_(mean[:,None,None])


def normalize(img, mean, std):
    mean, std = to_tensor_if_needed(mean).to(img.device), to_tensor_if_needed(std).to(img.device)
    return img.sub_(mean[:,None,None]).div_(std[:,None,None])