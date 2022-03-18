# --- built in ---
import functools
from typing import (
  Callable,
  Optional,
  Union
)
# --- 3rd party ---
import torch
from torch import nn
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.nets.base import DelayedModule

__all__ = [
  'Lambda',
  'Constant',
  'Swish'
]


# === Activ func ===

def swish(x: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
  """Swish activation function
  from arXiv:1710.05941
  
  Args:
    x (torch.Tensor): input tensors can be either 2D or 4D.
    beta (torch.Tensor): swish-beta can be a constant or a trainable params.
  
  Returns:
    torch.Tensor: output tensors
  """
  if len(x.size()) == 2:
    return x * torch.sigmoid(beta[None, :] * x)
  else:
    return x * torch.sigmoid(beta[None, :, None, None] * x)

def get_activ_fn(activ_fn: Union[str, Callable], dim: int) -> Callable:
  """Get activation function by name or function
  
  Args:
    activ_fn (Union[str, Callable]): activation function can be the name of the
      function or the function itself.
    dim (int): input dimensions.
  
  Returns:
    Callable: activation function
  """
  if isinstance(activ_fn, str):
    if rl_registry.get.activ(activ_fn) is not None:
      # get activation function from registry
      activ_class = rl_registry.get.activ(activ_fn)
      activ_fn = activ_class(dim)
    else:
      # get activation function from pure pytorch module
      return getattr(nn.functional, activ_fn)
  elif callable(activ_fn):
    return activ_fn
  else:
    raise ValueError(f"`activ_fn` must be a str or Callable, got {type(activ_fn)}")

# === Activation module ===

@rl_registry.register.activ('swish')
class Swish(DelayedModule):
  def __init__(self, dim: Optional[int]=None):
    """Swish activation function
    from arXiv:1710.05941

    Args:
        dim (int, optional): input dimensions. Defaults to -1.
    """
    super().__init__()
    if dim is not None:
      self.build(torch.Size([dim]))
  
  def build(self, input_shape: torch.Size):
    dim = input_shape[-1]
    self.beta = nn.Parameter(torch.ones(dim,))
    # ---
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return swish(x, self.beta)

class Activ(nn.Module):
  def __init__(
    self,
    dim: Optional[int] = None,
    activ: Union[str, Callable] = 'relu'
  ):
    """Custom activation layer

    Args:
        dim (int, optional): input dimensions. Used in `swish` activ fn.
          Defaults to -1.
        activ (Union[str, Callable], optional): activation function can be the
          name of the function or the function itself. Defaults to 'relu'.
    """
    super().__init__()
    self.activ = activ
    # ---
    if dim is not None:
      self.build(torch.Size([dim]))
  
  def build(self, input_shape: torch.Size):
    dim = input_shape[-1]
    self.activ_fn = get_activ_fn(self.activ, dim)
    # ---
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.activ_fn(x)

# === Other layers ===

class Lambda(nn.Module):
  def __init__(self, lambd: Callable):
    """Lambda layer

    Args:
      lambd (Callable): lambda function.
    """
    super().__init__()
    self.lambd = lambd
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.lambd(x)

class Constant(nn.Module):
  def __init__(
    self,
    constant: Union[torch.Tensor, nn.Parameter],
    requires_grad: bool = True
  ):
    """Constant layer
    This layer returns a user defined constant with the same shape
    as the input tensor passed into the `forward()`

    Args:
      constant (Union[torch.Tensor, nn.Parameter]): constant tensor.
      requires_grad (bool, optional): whether the constant requires
        gradient
    """
    super().__init__()
    # convert to tensor
    if not isinstance(constant, (torch.Tensor, nn.Parameter)):
      constant = rl_utils.to_tensor(constant)
    # convert tensor to nn.Parameter
    if isinstance(constant, torch.Tensor):
      constant = nn.Parameter(constant)
    # ensure type
    assert isinstance(constant, (torch.Tensor, nn.Parameter))
    self.constant = constant
    self.constant.requires_grad = bool(requires_grad)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # convert to the same dtype and device as `x`
    self.constant = self.constant.type_as(x)
    # broadcast shape to batch shape (b, *const.shape)
    shape = torch.broadcast_shapes(x.shape, self.const.shape)
    return torch.broadcast_to(self.constant, shape)
