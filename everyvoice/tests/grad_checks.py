"""Checks that inference code does not track gradients.

Two complementary checks:

- assert_no_param_grads: no parameter has accumulated a gradient.
- assert_no_grad_tracking: no output tensor is part of an autograd graph.

The second is the one that catches a missing ``torch.no_grad()``: ``.grad`` is
only filled by ``backward()``, so a forward pass that builds a graph still
leaves every ``param.grad`` as None, while the graph holds on to memory.
"""

from typing import Any

import torch


def assert_no_param_grads(model: torch.nn.Module, label: str = "model") -> None:
    """Assert that no parameter of model has a gradient."""
    with_grad = [
        name for name, param in model.named_parameters() if param.grad is not None
    ]
    assert not with_grad, f"{label}: parameters with a gradient: {with_grad}"


def _graph_tensors(value: Any, path: str) -> list[str]:
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            return [f"{path} (grad_fn={type(value.grad_fn).__name__})"]
        return []
    if isinstance(value, dict):
        return [
            found
            for key, item in value.items()
            for found in _graph_tensors(item, f"{path}[{key!r}]")
        ]
    if isinstance(value, (list, tuple)):
        return [
            found
            for i, item in enumerate(value)
            for found in _graph_tensors(item, f"{path}[{i}]")
        ]
    return []


def assert_no_grad_tracking(outputs: Any, label: str = "outputs") -> None:
    """Assert that no tensor in outputs (a tensor, or dicts/lists/tuples of
    them) requires a gradient or is attached to an autograd graph."""
    tracked = _graph_tensors(outputs, label)
    assert not tracked, f"tensors attached to an autograd graph: {tracked}"
