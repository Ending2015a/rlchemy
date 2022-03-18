# --- built in ---
from typing import (
  Any,
  Callable,
  List,
  Union,
  Tuple,
  Optional
)
# --- 3rd party ---
import gym
import numpy as np
import torch
from torch import nn
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib.nets import layers as rl_layers
from rlchemy.lib.nets.base import DelayedModule

__all__ = [
  'MlpNet',
  'NatureCnn',
  'AwesomeNet'
]

class MlpNet(DelayedModule):
  def __init__(
    self,
    dim: Optional[int] = None,
    mlp_units: List[int] = [64, 64],
    activ: Union[str, Callable] = 'relu'
  ):
    """Multi-layer perception (MLP) feature extractor

    Args:
      dim (int, optional): input dimension.
      mlp_units (List[int], optional): units of hidden layers.
        Defaults to [64, 64].
      activ (Union[str, Callable], optional): activation function.
        Defaults to 'relu'.
    """
    super().__init__()
    self.mlp_units: List[int] = mlp_units
    self.activ: str = activ
    # ---
    self.input_dim: Optional[int] = None
    self.output_dim: Optional[int] = None
    self._model: Optional[nn.Module] = None
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    """Build up module
    
    Note that this function is called automatically at the first `forward`
    call.

    Args:
      input_shape (torch.Size): input tensor shape. Expecting (..., dim),
        e.g. (b, dim) or (seq, b, dim). The last dimension is the input
        dimension size of the first linear layer. A multi-dimensional
        input is not allowed. e.g. image (batch, c, h, w). User should
        flatten the tensor to (batch, c*h*w) before forwarding to this
        module. Otherwise, the last dimension `w` is treated as the input
        dimension of the first linear layer.
    """
    self.input_dim = input_shape[-1]
    in_dim = self.input_dim
    layers = []
    for out_dim in self.mlp_units:
      layers.extend([
        nn.Linear(in_dim, out_dim),
        rl_layers.Activ(out_dim, activ=self.activ)
      ])
      in_dim = out_dim
    self._model = nn.Sequential(*layers)
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward MLP model"""
    return self._model(x)


class NatureCnn(DelayedModule):
  def __init__(
    self,
    shape: Tuple[int, ...] = None,
    activ: Union[str, Callable] = 'relu'
  ):
    """Nature CNN originated from paper
    "Human-level control through deep reinforcement learning"

    Args:
      shape: (Tuple[int, ...], optional): input shape must be a tuple of
        `int` with rank at least 3 indicating the shape of the image tensor
        (..., c, h, w), e.g. (b, c, h, w), (seq, b, c, h, w).
      activ (str, callable, optional): activation function.
        Defaults to 'relu'.
    """
    super().__init__()
    self.activ: str = activ
    # ---
    self.input_shape: Optional[torch.Size] = None
    self.input_dim: Optional[int] = None
    self.output_dim: Optional[int] = None
    self._model: Optional[nn.Module] = None
    if shape is not None:
      self.build(torch.Size(shape))

  def build(self, input_shape: torch.Size):
    """Build up module

    Args:
        input_shape (torch.Size): _
    """
    assert len(input_shape) >= 3, \
      f"The rank of `input_shape` must at least 3, got {len(input_shape)}"
    input_shape = input_shape[-3:]
    dim = input_shape[0]
    # create cnn
    cnn = nn.Sequential(
      nn.Conv2d(dim, 32, 8, 4, padding=0),
      rl_layers.Activ(32, activ=self.activ),
      nn.Conv2d(32, 64, 4, 2, padding=0),
      rl_layers.Activ(64, activ=self.activ),
      nn.Conv2d(64, 64, 3, 1, padding=0),
      rl_layers.Activ(64, activ=self.activ),
      nn.Flatten(start_dim=-3, end_dim=-1),
    )
    # forward cnn to get output size
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    outputs = cnn(dummy).detach()
    # append fc layers
    self._model = nn.Sequential(
      *cnn,
      nn.Linear(outputs.shape[-1], 512),
      rl_layers.Activ(512, activ=self.activ)
    )
    self.input_shape = input_shape
    self.input_dim = dim
    self.output_dim = 512
    self.mark_as_built()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward NatureCnn module"""
    return self._model(x)

# === Complex nets ===

class AwesomeNet(DelayedModule):
  def __init__(
    self,
    input_space: Optional[gym.spaces.Space] = None,
    force_mlp: bool = False,
    mlp_units: list = [64, 64],
    activ: str = 'relu'
  ):
    """AwesomeNet automatically handles nested spaces and input tensors.
    If the input space is a image space, AwesomeNet creates a NatureCnn
    for it, otherwise, it creates a MlpNet. If the input space is a nested
    space, AwesomeNet first flatten the spaces and creates NatureCnns for
    those image spaces and identity (flatten) layers for non image spaces
    as the feature extractors for each type of space. It then creates a
    single MlpNet as the final feature fusion layers.

    Args:
      input_space (gym.Space): input spaces that will forward into this net.
      force_mlp (bool, optional): force to use MlpNet as the feature 
        extractors. Defaults to False.
      mlp_units (list, optional): a list of numbers of hidden units for 
        each MlpNet layer. Defaults to [64, 64].
      activ (str, callable, optional): activation function.
        Defaults to 'relu'.
    """
    super().__init__()
    self.force_mlp: bool = force_mlp
    self.mlp_units: List[int] = mlp_units
    self.activ: str = activ
    # ---
    if input_space is not None:
      flat_spaces = rl_utils.flatten_space(input_space)
      assert len(flat_spaces) > 0
      self.build(flat_spaces)

  def get_input_shapes(
    self,
    x: Any,
    *args,
    **kwargs
  ) -> Any:
    """Get input tensor shapes
    The input tensor is expected to be a nested structure

    Args:
      x (Any): input tensor expecting a nested structure.

    Returns:
      Any: flattened tensor shapes
    """
    # flatten nested tensors to a tuple of tensors
    xs = rl_utils.flatten(x)
    return tuple([x.shape for x in xs])

  def build(
    self,
    input_shapes: Union[Tuple[gym.spaces.Space], Tuple[torch.Size]]
  ):
    assert len(input_shapes) > 0
    for shape in input_shapes:
      assert isinstance(shape, (torch.Size, gym.spaces.Space)), \
        f"Got an unexpected type of `shape`: {type(shape)}"
    # create models for each space. If it's an image space
    # use NatureCnn, otherwise, use Flatten
    models_and_dims = [
      self.create_model(shape)
      for shape in input_shapes
    ]
    models, dims = zip(*models_and_dims)
    self._models = nn.ModuleList(models)
    in_dim = np.sum(dims)
    # if there is only one image space, use identity layer as
    # the final output layer, otherwise (if there are multiple
    # spaces), use an extra mlp layers as the final output
    # layers to fuse features
    if (len(models) == 1 and isinstance(models, NatureCnn)):
      self._fuse_model = nn.Identity()
      out_dim = in_dim
    else:
      self._fuse_model, out_dim = self.create_mlp(in_dim)
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(self, x: Any, *args, **kwargs) -> torch.Tensor:
    """Forward networks"""
    xs = rl_utils.flatten(x)
    res = []
    for x, model in zip(xs, self._models):
      res.append(model(x))
    x = torch.cat(res, dim=-1)
    return self._fuse_model(x, *args, **kwargs)

  def create_model(
    self,
    shape: Union[torch.Size, gym.spaces.Space]
  ):
    """Create model by space

    Args:
      shape (Union[torch.Size, gym.spaces.Space]): input shape

    Returns:
      nn.Module: model
      int: output dimension
    """
    if isinstance(shape, gym.spaces.Space):
      space = shape
      use_cnn = (rl_utils.is_image_space(space)
        and not self.force_mlp)
      shape = space.shape
      rank = len(shape)
    else:
      shape = torch.Size(shape)
      # expecing a batched tensor shape
      # (..., b, c, h, w) or (..., b, dim)
      use_cnn = len(shape) > 4
      rank = 3 if use_cnn else 1
    if use_cnn:
      return self.create_cnn(shape)
    else:
      return nn.Flatten(-rank), np.prod(shape[-rank:])

  def create_cnn(self, shape):
    """Create CNN

    Args:
      dim (int): input dimension

    Returns:
      nn.Module: model
      int: output dimension
    """
    model = NatureCnn(shape, activ=self.activ)
    return model, model.output_dim

  def create_mlp(self, dim):
    """Create MLP

    Args:
      dim (int): input dimension

    Returns:
      nn.Module: model
      int: output dimension
    """
    model = MlpNet(dim, self.mlp_units, activ=self.activ)
    return model, model.output_dim
