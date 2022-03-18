# --- built in ---
import abc
import math
from typing import (
  Tuple
)
# --- 3rd party ---
import torch
from torch import nn
# --- my module ---
from rlchemy.lib.prob.distribs import Distribution
from rlchemy.lib import registry as rl_registry
from rlchemy.lib import utils as rl_utils

__all__ = [
  'Bijector',
  'Identity',
  'Tanh'
]

class Bijector(Distribution):
  def __init__(self, dist: Distribution, **kwargs):
    """Wrapping the base distribution with a bijector wrapper

    Args:
      dist (Distribution): Base distribution
    """
    if not isinstance(dist, Distribution):
      raise ValueError("`dist` must be a type of "
        f"Distribution, got {type(dist)}")
    super().__init__(dist.dtype, event_ndims=dist.event_ndims, **kwargs)
    self.dist = dist

  @property
  def shape(self) -> torch.Size:
    return self.dist.shape

  @property
  def device(self) -> torch.device:
    return self.dist.device

  @property
  def distribution(self) -> 'Distribution':
    return self.dist

  @property
  def root(self) -> 'Distribution':
    return self.dist.root

  @abc.abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward bijection

    Args:
      x (torch.Tensor): outcomes
    """
    raise NotImplementedError

  @abc.abstractmethod
  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    """Inverse bijection

    Args:
      y (torch.Tensor): outcomes
    """
    raise NotImplementedError

  @abc.abstractmethod
  def log_det_jacob(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward log-det-jacobian matrix if a given
    outcome.

    Args:
      x (torch.Tensor): outcomes
    """
    raise NotImplementedError

  def inject(self, x: torch.Tensor) -> torch.Tensor:
    """Forward this bijector, or forward all bijectors from root distribution.

    Args:
      x (torch.Tensor): outcomes before injecting to this bijector.
    """
    if hasattr(self.dist, 'inject'):
      x = self.dist.inject(x)
    return self.forward(x)
  
  def surject(self, y: torch.Tensor) -> torch.Tensor:
    """Backward this bijector, or all bijectors to root distribution.

    Args:
      y (torch.Tensor): outcomes before surjecting to this bijector.
      to_root (bool, optional): surject to the root distribution.
        Defaults to True.
    """
    y = self.inverse(y)
    if hasattr(self.dist, 'surject'):
      y = self.dist.surject(y)
    return y

  def mode(self) -> torch.Tensor:
    return self.forward(self.dist.mode())

  def sample(self, n: torch.Size=[]) -> torch.Tensor:
    """Sample outcomes"""
    return self.forward(self.dist.sample(n))

  def inject_log_prob(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute injected log likelihood from root outcomes.
    TODO: unittest

    Args:
      x (torch.Tensor): outcomes from the root distribution.

    Returns:
      torch.Tensor: injected outcomes
      torch.Tensor: injected log likelihood
    """
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    shape = torch.broadcast_shapes(x.shape, self.event_shape)
    x = torch.broadcast_to(x, shape)
    x, log_prob = self.dist.inject_log_prob(x)
    jacob = self.log_det_jacob(x)
    jacob = self.flatten_event_dims(jacob).sum(dim=-1)
    return self.forward(x), log_prob - jacob

  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    """Compute injected log likelihood from injected outcomes.
    TODO: unittest

    Args:
      x (torch.Tensor): injected outcomes or outcomes from the root
        distribution.

    Returns:
      torch.Tensor: log likelihood
    """
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    shape = torch.broadcast_shapes(x.shape, self.event_shape)
    x = torch.broadcast_to(x, shape)
    # compute injected log prob
    x, logp = self.inject_log_prob(self.surject(x))
    return logp

  def entropy(self) -> torch.Tensor:
    """Raw entropy"""
    return self.dist.entropy()

  def kl(self, q: Distribution) -> torch.Tensor:
    """Raw KL divergence"""
    return self.dist.kl(q)

@rl_registry.register.biject('identity')
class Identity(Bijector):
  """Identity bijector: y = x"""
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    return x

  def inverse(self, x: torch.Tensor) -> torch.Tensor:
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    return x

  def log_det_jacob(self, x: torch.Tensor) -> torch.Tensor:
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    return torch.zeros_like(x)

@rl_registry.register.biject('tanh')
class Tanh(Bijector):
  """Tanh bijector: y= tanh(x)"""
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    return torch.tanh(x)

  def inverse(self, y: torch.Tensor) -> torch.Tensor:
    y = rl_utils.to_tensor(y, dtype=self.dtype, device=self.device)
    return self.atanh(y)

  @staticmethod
  def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x.log1p() - (-x).log1p())

  def log_det_jacob(self, x: torch.Tensor) -> torch.Tensor:
    """Compute the forward log-det-jacobian matrix if a given outcome.
    This is equivalent to
      log(dy/dx) = log(1 - tanh(x)**2)
    where y = tanh(x)
    TODO: unittest

    Args:
      x (torch.Tensor): outcomes
    """
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    # log(dy/dx) = log(1 - tanh(x)**2)
    # more numerically stable implementation
    return 2. * (math.log(2.) - x - nn.functional.softplus(-2. * x))
  
  def entropy(self) -> torch.Tensor:
    """Raw entropy without correction
    Entropy for Tanh bijector does not exist an analytical solution.
    """
    return self.dist.entropy()
    
  def kl(self, q: Distribution) -> torch.Tensor:
    """Raw KL divergence without correction
    KL-divergence for Tanh bijector does not exist an analytical solution.
    """
    return self.dist.kl(q)