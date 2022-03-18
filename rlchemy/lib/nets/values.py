# --- built in ---
import collections
from typing import (
  Optional,
  List,
  Tuple,
  Union
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
# --- my module ---
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.nets.base import DelayedModule, Parallel

__all__ = [
  'ValueNet',
  'MultiHeadValueNets'
]

class ValueNet(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    out_dim: int = 1
  ):
    """Base value net

    Args:
      dim (int): input dimension.
      out_dim (int, optional): output dimension. Defaults to 1.
      net (nn.Module, optional): base network, feature extractor.
        Defaults to None.
    """
    super().__init__()
    self.out_dim = out_dim
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      self.build([dim])
  
  def build(self, input_shape: torch.Size):
    in_dim = input_shape[-1]
    out_dim = self.out_dim
    self._model = self.make_model(in_dim, out_dim)
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward value net

    Args:
      x (torch.Tensor): input tensor in shape (b, *).

    Returns:
      torch.Tensor: value predictions, (Default) shape (b, out_dim)
    """
    return self._model(x)

  def make_model(self, dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(dim, out_dim)


class MultiHeadValueNets(DelayedModule):
  def __init__(
    self,
    dims: Optional[List[int]] = None,
    out_dims: Union[int, List[int]] = 1,
    n_heads: int = 1
  ):
    """Base multi head value net

    Args:
      dims (int, list): input dimensions.
      out_dims (int, list, optional): expected output dimensions.
        Defaults to 1.
      n_heads (int, optional): number of heads. Defaults to 1.
    """
    super().__init__()
    self.out_dims = out_dims
    self.n_heads = n_heads
    # ---
    self.input_dims = None
    self.output_dims = None
    self._model = None
    if dims is not None:
      if isinstance(dims, collections.Iterable):
        self.build(
          tuple(torch.Size([dim]) for dim in dims)
        )
      else:
        # single scalar
        self.build(torch.Size([dims]))

  def get_input_shapes(
    self,
    x: Union[torch.Tensor, Tuple[torch.Tensor]],
    *args,
    **kwargs
  ) -> Union[torch.Size, Tuple[torch.Size]]:
    if torch.is_tensor(x):
      return x.shape
    else:
      xs = x
      return tuple(x.shape for x in xs)

  def build(
    self,
    input_shapes: Union[torch.Size, Tuple[torch.Size]]
  ):
    if isinstance(input_shapes, torch.Size):
      input_shapes = tuple([input_shapes] * self.n_heads)
    if not isinstance(out_dims, collections.Iterable):
      out_dims = tuple([out_dims] * self.n_heads)
    
    assert len(input_shapes) == self.n_heads
    assert len(out_dims) == self.n_heads

    # [(model, in_dim, out_dim), ...]
    models_and_dims = [
      (
        self.make_model(input_shapes[n][-1], out_dims[n]),
        input_shapes[n][-1],
        out_dims[n]
      ) for n in range(self.n_heads)
    ]
    models, in_dims, out_dims = zip(*models_and_dims)
    self._models = nn.ModuleList(models)
    self.input_dims = in_dims
    self.output_dims = out_dims
    self.mark_as_built()

  def forward(
    self,
    x: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    *args,
    **kwargs
  ) -> Tuple[torch.Tensor, ...]:
    """Forward all value nets or specified value nets
    if `indices` is not None

    Args:
      x (torch.Tensor): input tensor in shape (b, latent), torch.float32

    Returns:
      Tuple[torch.Tensor, ...]: value predictions with shapes (b, out_dim)
    """
    if not torch.is_tensor(x) and len(x) == self.n_heads:
      # forward tensors into heads one-by-one
      xs = x
      return tuple(
        model(x) for x, model in zip(xs, self._models)
      )
    else:
      # single tensor
      return tuple(
        model(x) for model in self._models
      )

  def make_model(self, dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(dim, out_dim)
