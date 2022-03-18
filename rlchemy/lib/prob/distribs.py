# --- built in ---
import abc
import math
from typing import (
  Tuple,
  Union
)
# --- 3rd party ---
import numpy as np
import torch
from torch import nn
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import registry as rl_registry

__all__ = [
  'Distribution',
  'Categorical',
  'Normal',
  'MultiNormal'
]

class Distribution(nn.Module, metaclass=abc.ABCMeta):
  def __init__(self, dtype: torch.dtype, event_ndims: int=0):
    """Distribution base class

    Args:
      dtype (torch.dtype): type of outcomes.
      event_ndims (int, optional): Number of event dimensions. Defaults to 0.
    """
    super().__init__()
    self.dtype = dtype
    self.event_ndims = event_ndims

  @property
  @abc.abstractmethod
  def shape(self) -> torch.Size:
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def device(self) -> torch.device:
    raise NotImplementedError

  @property
  def root(self) -> 'Distribution':
    return self

  @property
  def event_shape(self) -> torch.Size:
    if self.event_ndims == 0:
      return torch.Size([])
    return self.shape[-self.event_ndims:]

  @property
  def batch_shape(self) -> torch.Size:
    if self.event_ndims == 0:
      return self.shape
    return self.shape[:-self.event_ndims]

  def flatten_event_dims(self, x: torch.Tensor) -> torch.Tensor:
    """Flatten the event dimensions of 'x'.
    The flattened 'x' has at least one event dimension.

    For example:
    ```
    a = torch.zeros((2, 3, 4))
    # (2, 12)
    Distribution(event_ndims=2).flatten_event_ndims(a).shape
    b = torch.zeros((4,))
    # AssertionError
    Distribution(event_ndims=2).flatten_event_ndims(b).shape
    c = torch.zeros((2, 3, 4))
    # (2, 3, 4, 1)
    Distribution(event_ndims=0).flatten_event_ndims(c).shape
    ```

    Args:
      x (torch.Tensor): input tensor

    Returns:
      torch.Tensor: flattened tensor
    """
    assert len(x.shape) >= self.event_ndims, \
      "The rank of tensor 'x' must be greater than event rank, " \
      f"got {len(x.shape)} vs {self.event_ndims}"
    if self.event_ndims >= 0:
      shape = x.shape[:-self.event_ndims]
      x = x.view(*shape, -1)
    return x

  def flatten_batch_dims(self, x: torch.Tensor) -> torch.Tensor:
    """Flatten the batch dimensions of 'x'.
    The flattened 'x' has at least one batch dimension.

    For example:
    ```
    a = torch.zeros((2, 3, 4))
    # (6, 4)
    Distribution(event_ndims=1).flatten_batch_dims(a).shape
    b = torch.zeros((2, 3, 4))
    # (1, 2, 3, 4)
    Distribution(event_ndims=3).flatten_batch_dims(b).shape
    ```

    Args:
      x (torch.Tensor): input tensor

    Returns:
      torch.Tensor: flattened tensor
    """
    assert len(x.shape) >= self.event_ndims, \
      "The rank of tensor 'x' must be greater than event rank, " \
      f"got {len(x.shape)} vs {self.event_ndims}"
    if len(x.shape) - self.event_ndims >= 0:
      shape = x.shape[-self.event_ndims:]
      x = x.view(-1, *shape)
    return x

  def prob(self, x: torch.Tensor) -> torch.Tensor:
    """Probability of given outcomes (x)"""
    return torch.exp(self.log_prob(x))

  def inject_log_prob(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return outcomes and log probabilities"""
    return x, self.log_prob(x)

  @abc.abstractmethod
  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    """Probability of given outcomes (x)"""
    raise NotImplementedError

  @abc.abstractmethod
  def mode(self) -> torch.Tensor:
    """Mode"""
    raise NotImplementedError

  @abc.abstractmethod
  def sample(self, n: Union[int, torch.Size]=[]) -> torch.Tensor:
    """Sample outcomes"""
    raise NotImplementedError

  @abc.abstractmethod
  def entropy(self) -> torch.Tensor:
    """Entropy"""
    raise NotImplementedError

  @abc.abstractmethod
  def kl(self, q: 'Distribution') -> torch.Tensor:
    """KL divergence

    Args:
      q (Distribution): target probability ditribution.
    """
    raise NotImplementedError

@rl_registry.register.distrib('categorical')
class Categorical(Distribution):
  def __init__(
    self,
    logits: torch.Tensor,
    dtype: torch.dtype=torch.int32,
    **kwargs
  ):
    """Categorical distribution

    Args:
      logits (torch.Tensor): logits.
      dtype (torch.dtype, optional): outcome type. Defaults to torch.int32.
    """
    kwargs.pop('event_ndims', None)
    super().__init__(dtype=dtype, event_ndims=1, **kwargs)
    logits = rl_utils.to_tensor(logits)
    self._logits = logits
    logits = self._logits - torch.amax(self._logits, dim=-1, keepdim=True)
    self._norm_logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

  @property
  def logits(self) -> torch.Tensor:
    return self._logits

  @property
  def shape(self) -> torch.Size:
    return self._logits.shape

  @property
  def device(self) -> torch.device:
    return self._logits.device

  def probs(self) -> torch.Tensor:
    """Probability distribution (pdf)"""
    return torch.exp(self.log_probs())

  def log_probs(self) -> torch.Tensor:
    """Log probability distribution (log pdf)"""
    return self._norm_logits

  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    """Log probability of given outcomes (x)
    
    Args:
      x (torch.Tensor): outcome, torch.int32
    """
    logits = self.flatten_batch_dims(self.logits)
    labels = rl_utils.to_tensor(x, dtype=torch.int64, device=self.device)
    labels = labels.view(-1)
    return -nn.functional.cross_entropy(
      logits, labels, reduction='none').view(self.batch_shape)

  def mode(self) -> torch.Tensor:
    """Mode"""
    return torch.argmax(self.logits, dim=-1).to(dtype=self.dtype)

  def sample(self, n: Union[int, torch.Size]=[]) -> torch.Tensor:
    """Sample outcomes"""
    shape = (*np.atleast_1d(n), *self.shape) # concat shape
    e  = torch.rand(shape, dtype=self.logits.dtype, device=self.device)
    it = self.logits - torch.log(-torch.log(e))
    return torch.argmax(it, dim=-1).to(dtype=self.dtype)

  def entropy(self) -> torch.Tensor:
    """Entropy"""
    eps = torch.finfo(self.logits.dtype).min
    p = torch.exp(self.log_probs())
    logp = torch.clamp(self.log_probs(), min=eps)
    return torch.sum(-p * logp, dim=-1)
    # m = torch.max(self.logits, dim=-1, keepdim=True)[0]
    # x = self.logits - m
    # z = torch.sum(torch.exp(x), dim=-1)
    # y = torch.exp(x)
    # e = torch.where(y == 0., y, self.logits * y) # avoid nan
    # p = torch.sum(e, dim=-1) / z
    # return m[..., 0] + torch.log(z) - p

  def kl(self, q: 'Categorical') -> torch.Tensor:
    """KL divergence

    Args:
      q (Categorical): target probability distribution.
    """
    logp = self.log_probs()
    logq = q.log_probs()
    p = torch.exp(logp)
    return torch.sum(p * (logp - logq), dim=-1)

@rl_registry.register.distrib('normal')
class Normal(Distribution):
  def __init__(
    self,
    mean: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    **kwargs
  ):
    """Gaussian normal distribution

    Args:
      mean (torch.Tensor): gaussian mean/loc
      scale (torch.Tensor): gaussian scale/std
      dtype (torch.dtype, optional): outcome type. Defaults to torch.float32.
    """
    kwargs.pop('event_ndims', None)
    super().__init__(dtype=dtype, event_ndims=0, **kwargs)
    self._mean = rl_utils.to_tensor(mean, dtype=dtype)
    self._scale = rl_utils.to_tensor(scale, dtype=dtype)
    self._shape = torch.broadcast_shapes(self._mean.shape, self._scale.shape)

  @property
  def mean(self) -> torch.Tensor:
    return self._mean

  @property
  def scale(self) -> torch.Tensor:
    return self._scale

  @property
  def shape(self) -> torch.Size:
    return self._shape

  @property
  def device(self) -> torch.device:
    return self._mean.device

  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    """Log probability
    
    Args:
      x (torch.Tensor): outcomes, torch.float32
    """
    x = rl_utils.to_tensor(x, dtype=self.dtype, device=self.device)
    var = (self.scale ** 2.0)
    log_scale = self.scale.log()
    z = math.log(math.sqrt(2 * math.pi))
    return -((x - self.mean) ** 2) / (2 * var) - log_scale - z

  def mode(self) -> torch.Tensor:
    """Mode"""
    return self.mean * torch.ones_like(self.scale)

  def sample(self, n: Union[int, torch.Size]=[]) -> torch.Tensor:
    """Sample outcomes"""
    shape = (*np.atleast_1d(n), *self.shape) # concat shape
    x = torch.randn(shape, device=self.device).to(dtype=self.dtype)
    return self.mean + x * self.scale

  def entropy(self) -> torch.Tensor:
    """Entropy"""
    c = 0.5 * math.log(2. * math.pi) + 0.5
    return c + torch.log(self.scale) * torch.ones_like(self.mean)

  def kl(self, q: 'Normal') -> torch.Tensor:
    """KL divergence
    
    Args:
      q (Normal): target probability distribution
    """
    log_diff = (torch.log(self.scale * torch.ones_like(self.mean))
          - torch.log(q.scale * torch.ones_like(q.mean)))
    return (0.5 * (self.mean/q.scale - q.mean/q.scale)**2.0 +
        0.5 * torch.expm1(2. * log_diff) - log_diff)

@rl_registry.register.distrib('multi_normal')
class MultiNormal(Normal):
  def __init__(
    self,
    mean: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    event_ndims: int = 1,
    **kwargs
  ):
    """Multivariate gaussian distribution

    Args:
      mean (torch.Tensor): gaussian mean/loc
      scale (torch.Tensor): gaussian scale/std
      dtype (torch.dtype, optional): outcome type. Defaults to torch.float32.
      event_ndims (int, optional): number of event dimensions. Defaults to 1.
    """
    Distribution.__init__(
      self,
      dtype = dtype,
      event_ndims = event_ndims,
      **kwargs
    )
    assert event_ndims >= 1, \
      ("MultiNormal distribution has at least one "
      f"event dim, got {event_ndims}")
    self._mean = rl_utils.to_tensor(mean, dtype=dtype)
    self._scale = rl_utils.to_tensor(scale, dtype=dtype)
    self._shape = torch.broadcast_shapes(self._mean.shape, self._scale.shape)
    if len(self._shape) < 1:
      raise RuntimeError('MultiNormal needs at least 1 dimension')

  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    """Log probability"""
    return self.flatten_event_dims(super().log_prob(x)).sum(dim=-1)

  def entropy(self) -> torch.Tensor:
    """Entropy"""
    return self.flatten_event_dims(super().entropy()).sum(dim=-1)

  def kl(self, q: 'MultiNormal') -> torch.Tensor:
    """KL divergence

    Args:
      q (MultiNormal): target probability distribution.
    """
    return self.flatten_event_dims(super().kl(q)).sum(dim=-1)
