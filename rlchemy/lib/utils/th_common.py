# --- built in ---
import random
from typing import (
  Any,
  Optional,
  List,
  Union
)
# --- 3rd party ---
import gym
import torch
from torch import nn
import numpy as np
# --- my module ---
from rlchemy.lib.utils import common as rl_common
from rlchemy.lib.utils import nest as rl_nest

__all__ = [
  'set_seed',
  'to_numpy',
  'to_tensor',
  'to_tensor_like',
  'normalize',
  'denormalize',
  'preprocess_obs',
  'soft_update',
  'input_tensor'
]

def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

def to_numpy(
  inputs: Any,
  dtype: Optional[np.dtype] = None
) -> np.ndarray:
  """Convert inputs to numpy array

  Args:
    inputs (Any): input tensor
    dtype (np.dtype, optional): data type. Defaults to None.

  Returns:
    np.ndarray: numpy data
  """
  if torch.is_tensor(inputs):
    t = inputs.detach().cpu().numpy()
  else:
    t = np.asarray(inputs)
  dtype = dtype or t.dtype
  return t.astype(dtype=dtype)

def to_tensor(
  inputs: Any,
  dtype: Optional[torch.dtype] = None,
  device: Optional[torch.device] = None,
  **kwargs
) -> torch.Tensor:
  """Convert inputs to torch.Tensor

  Args:
    inputs (Any): input tensor.
    dtype (torch.dtype, optional): torch dtype. Defaults to None.
    device (torch.device, optional): torch device. Defaults to None.

  Returns:
    torch.Tensor: torch tensor
  """
  if torch.is_tensor(inputs):
    t = inputs
  elif isinstance(inputs, np.ndarray):
    t = torch.from_numpy(inputs)
  else:
    t = torch.tensor(inputs, dtype=dtype)
  return t.to(device=device, dtype=dtype, **kwargs)

def to_tensor_like(inputs: Any, x: torch.Tensor) -> torch.Tensor:
  """Convert `inputs` to torch.Tensor with the same dtype and
  device as `x`

  Args:
    inputs (Any): input tensor
    x (torch.Tensor): torch tensor

  Returns:
    torch.Tensor: torch tensor
  """
  assert torch.is_tensor(x), f"`x` must be a torch.Tensor, got {type(x)}"
  return to_tensor(inputs, dtype=x.dtype, device=x.device)

def normalize(
  x: Union[float, np.ndarray, torch.Tensor],
  low: Union[float, np.ndarray, torch.Tensor],
  high: Union[float, np.ndarray, torch.Tensor],
  nlow: Union[float, np.ndarray, torch.Tensor] = 0.0,
  nhigh: Union[float, np.ndarray, torch.Tensor] = 1.0
) -> Union[float, np.ndarray, torch.Tensor]:
  """Normalize x from [low, high] to [nlow, nhigh]"""
  if torch.is_tensor(x):
    low = to_tensor_like(low, x)
    high = to_tensor_like(high, x)
    nlow = to_tensor_like(nlow, x)
    nhigh = to_tensor_like(nhigh, x)
  return ((nhigh-nlow)/(high-low)) * (x-low) + nlow

def denormalize(
  x: Union[float, np.ndarray, torch.Tensor],
  low: Union[float, np.ndarray, torch.Tensor],
  high: Union[float, np.ndarray, torch.Tensor],
  nlow: Union[float, np.ndarray, torch.Tensor] = 0.0,
  nhigh: Union[float, np.ndarray, torch.Tensor] = 1.0
) -> Union[float, np.ndarray, torch.Tensor]:
  """Denormalize x from [nlow, nhigh] to [low, high]"""
  if torch.is_tensor(x):
    low = to_tensor_like(low, x)
    high = to_tensor_like(high, x)
    nlow = to_tensor_like(nlow, x)
    nhigh = to_tensor_like(nhigh, x)
  return ((high-low)/(nhigh-nlow)) * (x-nlow) + low

def preprocess_obs(
  inputs: Union[np.ndarray, torch.Tensor],
  obs_space: gym.spaces.Space,
  dtype: torch.dtype = torch.float32,
  device: torch.device = None,
  low_tensor: Optional[torch.Tensor] = None,
  high_tensor: Optional[torch.Tensor] = None
) -> Any:
  """Preprocess non-float observations
  If the input's dtype is not float32 or float64 we normalize it.
  TODO: support gym.spaces.Tuple and gym.spaces.Dict

  Args:
    inputs (Union[np.ndarray, torch.Tensor]): observation samples.
    obs_space (gym.spaces.Space): observation space.
    dtype (torch.dtype, optional): tensor data type. Defaults to
      torch.float32.
    device (torch.device, optional): target device. Defaults
      to None.
    low_tensor (torch.Tensor, optional): cached lower bound
      tensors for `gym.spaces.Box`.
    high_tensor (torch.Tensor, optional): cached upper bound
      tensors for `gym.spaces.Box`.

  Returns:
    Any: preprocessed observations in `dtype` stored on
      `device`
  """
  inputs = to_tensor(inputs)
  # Do nothing if input is a float
  if inputs.dtype in [torch.float32, torch.float64]:
    return inputs.to(dtype=dtype, device=device)
  # Normalize
  if isinstance(obs_space, gym.spaces.Box):
    inputs = inputs.to(dtype=dtype)
    if rl_common.is_image_space(obs_space) and np.all(obs_space.high == 255):
      inputs = inputs / 255.
    elif rl_common.is_bounded(obs_space):
      if low_tensor is None:
        low_tensor = obs_space.low
      if high_tensor is None:
        high_tensor = obs_space.high
      low = to_tensor_like(low_tensor, inputs)
      high = to_tensor_like(high_tensor, inputs)
      inputs = normalize(inputs, low, high, 0., 1.)
  elif isinstance(obs_space, gym.spaces.Discrete):
    inputs = nn.functional.one_hot(inputs.long(), num_classes=obs_space.n)
  elif isinstance(obs_space, gym.spaces.MultiDiscrete):
    # inputs = [3, 5] obs_space.nvec = [4, 7]
    # [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    nvec = obs_space.nvec
    inputs = torch.cat([
      nn.functional.one_hot(inputs[..., idx], num_classes=nvec[idx])
      for idx in range(inputs.shape[-1])
    ], dim=-1)
  elif isinstance(obs_space, gym.spaces.MultiBinary):
    pass
  else:
    raise NotImplementedError("Preprocessing not implemented for "
      f"{type(obs_space)}")
  return inputs.to(dtype=dtype, device=device)

def soft_update(
  target_vars: List[nn.Parameter],
  source_vars: List[nn.Parameter],
  tau: float = 0.005
):
  """Perform soft updates

  Args:
    target_vars (List[nn.Parameter]): a list of torch.Parameter update to.
    source_vars (List[nn.Parameter]): a list of torch.Parameter update from.
    tau (float, optional): smooth rate. Defaults to 0.005.
  """
  target_vars = list(target_vars)
  source_vars = list(source_vars)
  if len(target_vars) != len(source_vars):
    raise ValueError("The length of parameter lists does not match, "
      f"got {len(target_vars)} vs {len(source_vars)}")
  with torch.no_grad():
    for tar, src in zip(target_vars, source_vars):
      tar.data.mul_(1 - tau)
      torch.add(tar.data, src.data, alpha=tau, out=tar.data)

def input_tensor(space=None, shape=None, batch_size=1, dtype=None):
    """Create zero tensors either from
    1. `space` if it's given. In this case, the `shape` is ignored. If `dtype`
        is not provided, use `space.dtype`.
    2. `shape` if `space` is not given. `dtype` defaults to `torch.float32`.

    Args:
        space (gym.Space): gym space. support nested type (Dict, Tuple).
        shape (list, tuple, optional): shape of zero tensors to create.
            Defaults to None.
        batch_size (int, optional): size of batch dim. Defaults to 1.
        dtype (torch.dtype, optional): tensor type. Defaults to None.

    Returns:
        torch.Tensor: zero tensors.
    """
    if space is not None:
        # create tensors from the given space.
        _tensor_op = lambda sp: (to_tensor(
          np.zeros(
            (batch_size, *sp.shape),
            dtype=sp.dtype
          ), dtype=dtype
        )) # ignored if dtype is None.
        return rl_nest.map_nested_space(space, _tensor_op)
    else:
        # create tensors from the given shape and dtype.
        assert isinstance(shape, (list, tuple, torch.Size))
        dtype = dtype or torch.float32
        return to_tensor(np.zeros((batch_size, *shape)), dtype=dtype)
