import torch


def _torch_compile_warmup(model: torch.nn.Module, warmup_tensor: torch.Tensor) -> None:
    """Warm up the model by passing in a tensor multiple times.

    Args:
        model: The PyTorch model to warm up.
        warmup_tensor: The tensor to pass through the model.
    """
    # warm up the model by passing in zeros multiple times
    # even the first couple of times are slow due to the reduce-overhead mode, so we have multiple warmup calls
    # see: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    model.eval()
    with torch.no_grad():
        model(warmup_tensor)
        model(warmup_tensor)
        model(warmup_tensor)
