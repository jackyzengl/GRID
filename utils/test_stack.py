import torch

def stack_tensors(data):
    if isinstance(data, dict):
        # Recursively call for dictionary values
        return {key: stack_tensors(value) for key, value in data.items()}
    elif torch.is_tensor(data): 
        # Base case - stack tensor 
        return torch.stack(data)
    else:
        return data