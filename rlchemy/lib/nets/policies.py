# --- built in ---
from typing import (
  Optional,
  Union
)
# --- 3rd party ---
import gym
import numpy as np
import torch
from torch import nn
# --- my module ---
from rlchemy.lib import prob as rl_prob
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.nets.base import DelayedModule

__all__ = [
  'CategoricalPolicyNet',
  'DiagGaussianPolicyNet',
  'PolicyNet',
]


class CategoricalPolicyNet(DelayedModule):
  support_spaces = [gym.spaces.Discrete]
  def __init__(
    self,
    dim: Optional[int] = None,
    action_space: gym.spaces.Space = None
  ):
    """Categorical policy for discrete action space

    Args:
      dim (int): input dimension
      action_space (gym.Space): action space, must be
        one of the types listed in `support_spaces`
    """
    super().__init__()
    assert action_space is not None
    if not isinstance(action_space, tuple(self.support_spaces)):
      raise ValueError(f"{type(self)} does not support "
        f"action spaces of type `{type(action_space)}`")
    self.action_space = action_space
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    in_dim = input_shape[-1]
    out_dim = self.action_space.n
    self._model = self.make_model(in_dim, out_dim)
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(
    self,
    x: torch.Tensor,
    *args,
    **kwargs
  ) -> rl_prob.Categorical:
    """Forward network

    Args:
      x (torch.Tensor): input tensor in shape (b, latent), torch.float32

    Returns:
      Categorical: A categorical distribution
    """
    return rl_prob.Categorical(self._model(x, *args, **kwargs))

  def make_model(self, dim: int, out_dim: int) -> nn.Module:
    """Create policy model"""
    return nn.Linear(dim, out_dim)

class DiagGaussianPolicyNet(DelayedModule):
  support_spaces = [gym.spaces.Box]
  def __init__(
    self,
    dim: Optional[int] = None,
    action_space: gym.spaces.Space = None,
    squash: bool = False
  ):
    """Tanh squashed diagonal Gaussian mixture policy net
    with state-dependent covariance

    Args:
      dim (int): input dimension
      action_space (gym.Space): action space, must be
        one of the types listed in `support_spaces`
      squash (bool, optional): apply tanh squashing. Defaults to False.
    """
    super().__init__()
    assert action_space is not None
    if not isinstance(action_space, tuple(self.support_spaces)):
      raise ValueError(f"{type(self).__name__} does not "
        f"suprt action space of type `{type(action_space)}`")
    self.action_space = action_space
    self.squash = squash
    self._event_ndims = len(self.action_space.shape)
    # ---
    self.input_dim = None
    self.output_dim = None
    self._mean_model = None
    self._logstd_model = None
    if dim is not None:
      self.build(torch.Size([dim]))

  def build(self, input_shape: torch.Size):
    in_dim = input_shape[-1]
    out_dim = int(np.prod(self.action_space.shape))
    self._mean_model = self.make_mean_model(in_dim, out_dim)
    self._std_model = self.make_logstd_model(in_dim, out_dim)
    self.input_dim = in_dim
    self.output_dim = out_dim
    self.mark_as_built()

  def forward(
    self,
    x: torch.Tensor,
    *args,
    **kwargs
  ) -> rl_prob.MultiNormal:
    """Forward network
    #TODO handle *args, **kwargs

    Args:
      x (torch.Tensor): input tensor in shape (b, latent), torch.float32

    Returns:
      MultiNormal: multi variate gaussian distribution
    """
    # forward model
    mean = self._mean_model(x)
    logstd = self._logstd_model(x)
    std = torch.exp(logstd)
    # reshape as action space shape (-1 is the batch dim)
    mean = mean.view(-1, *self.action_space.shape)
    std = std.view(-1, *self.action_space.shape)
    # create multi variate gauss dist with tanh squashed
    dist = rl_prob.MultiNormal(mean, std,
        event_ndims=self._event_ndims)
    if self.squash:
      dist = rl_prob.Tanh(dist)
    else:
      dist = rl_prob.Identity(dist)
    return dist

  def make_mean_model(self, dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(dim, out_dim)

  def make_logstd_model(self, dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(dim, out_dim)

class PolicyNet(DelayedModule):
  # TODO: support other type of spaces: MultiDiscrete, Dict, Tuple
  support_spaces = [gym.spaces.Box, gym.spaces.Discrete]
  def __init__(
    self,
    dim: Optional[int] = None,
    action_space: gym.spaces.Space = None,
    squash: bool = False
  ):
    """Base Policy net

    Args:
      dim (int): input dimension.
      action_space (gym.Space): action space, must be
        one of the types listed in `support_spaces`
      squash (bool, optional): apply tanh squash for continuous (Box)
        action space. Defaults to False.
    """
    super().__init__()
    # check if space supported
    assert action_space is not None
    if not isinstance(action_space, tuple(self.support_spaces)):
      raise ValueError(f"{type(self)} does not "
        f"support the action space of type {type(action_space)}")
    self.action_space = action_space
    self.squash = squash
    # ---
    self.input_dim = None
    self.output_dim = None
    self._model = None
    if dim is not None:
      self.build(torch.Size([dim]))
    
  def build(self, input_shape: torch.Size):
    in_dim = input_shape[-1]
    if isinstance(self.action_space, gym.spaces.Discrete):
      model = self.make_categorical_policy(in_dim)
    elif isinstance(self.action_space, gym.spaces.Box):
      model = self.make_gaussian_policy(in_dim)
    else:
      raise ValueError(f"{type(self)} dose not "
        f"support the action space of type {type(self.action_space)}")
    self._model = model
    self.input_dim = in_dim
    self.output_dim = model.output_dim
    self.mark_as_built()

  def forward(self, x, *args, **kwargs):
    """Forward network

    Args:
      x (torch.Tensor): input tensor in shape (b, *).

    Returns:
      Distribution: action distributions.
    """
    return self._model(x, *args, **kwargs)

  def make_categorical_policy(self, dim):
    return CategoricalPolicyNet(dim, self.action_space)

  def make_gaussian_policy(self, dim):
    return DiagGaussianPolicyNet(dim, self.action_space, self.squash)