# --- built n
import abc
from typing import (
  Any,
  List,
  Optional,
  Tuple,
  Union
)
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils

__all__ = [
  'compute_nstep_rew',
  'compute_advantage',
  'SegmentTree',
  'RelativeIndex'
]

def compute_nstep_rew(
  rew: np.ndarray,
  done: np.ndarray,
  gamma: float = 0.99
) -> np.ndarray:
  """Compute N step rewards
  aggregate the folloing N-1 steps discounted rewards to the 
  first reward:
  rew = rew[0] + gamma**1 * rew[1] + gamma**2 * rew[2] + ...

  Args:
    rew (np.ndarray): Reward sequences, (step, ...)
    done (np.ndarray): Done sequences, (step, ...)
    gamma (float, optional): Discount factor. Defaults to 0.99

  Returns:
    np.ndarray: N step rewards (...)
  """
  rew = rl_utils.to_numpy(rew).astype(np.float32)
  done = rl_utils.to_numpy(done).astype(np.float32)
  mask = 1.-done
  if len(done) == 0:
    return rew
  return _compute_nstep_rew(rew=rew, mask=mask, gamma=gamma)

def _compute_nstep_rew(
  rew: np.ndarray,
  mask: np.ndarray,
  gamma: float = 0.99
) -> np.ndarray:
  gams = gamma ** np.arange(len(mask))
  res = np.zeros_like(rew[0])
  prev_mask = np.ones_like(mask[0])
  for t in range(len(mask)):
    res += gams[t] * rew[t] * prev_mask
    prev_mask *= mask[t]
  return res

def compute_advantage(
  rew: np.ndarray,
  val: np.ndarray,
  done: np.ndarray,
  gamma: float = 0.99,
  gae_lambda: float = 1.0
) -> np.ndarray:
  """Compute GAE

  Args:
    rewards (np.ndarray): Rewards (steps, ...)
    values (np.ndarray): Predicted values (steps, ...)
    dones (np.ndarray): Done flags (steps, ...)
    gamma (float, optional): Discount factor. Defaults to 0.99.
    gae_lambda (float, optional): GAE lambda. Defaults to 1.0.

  Returns:
    np.ndarray: GAE
  """
  # ensure the inputs are np.ndarray
  rew  = rl_utils.to_numpy(rew).astype(np.float32)
  val  = rl_utils.to_numpy(val).astype(np.float32)
  done = rl_utils.to_numpy(done).astype(np.float32)
  mask = 1.-done
  return _compute_advantage(rew=rew, val=val, mask=mask,
    gamma=gamma, gae_lambda=gae_lambda)

def _compute_advantage(
  rew: np.ndarray,
  val: np.ndarray,
  mask: np.ndarray,
  gamma: float,
  gae_lambda: float
) -> np.ndarray:
  """Perform GAE computation"""
  adv = np.zeros_like(rew)
  gae = 0.
  next_mask = mask[-1]
  next_val  = val[-1]
  for t in reversed(range(len(mask))):
    delta = rew[t] + gamma * next_val * next_mask - val[t]
    gae = delta + gamma * gae_lambda * next_mask * gae
    adv[t] = gae
    next_mask = mask[t]
    next_val  = val[t]
  return adv

class SegmentTree(metaclass=abc.ABCMeta):
  def __init__(self, size: int):
    """An implementation of segment tree used to efficiently O(logN)
    compute the sum of a query range [start, end)

    Args:
      size (int): Number of elements.
    """
    assert isinstance(size, int) and size > 0
    base = 1<<(size-1).bit_length()
    self._size = size
    self._base = base
    self._value = np.zeros([base * 2], dtype=np.float64)
  
  def __getitem__(self, key: np.ndarray) -> np.ndarray:
    # formalize indices
    if isinstance(key, (int, slice)):
      key = np.asarray(range(self._size)[key], dtype=np.int64)
    else:
      key = np.asarray(key, dtype=np.int64)
    key = key % self._size + self._base
    return self._value[key]

  def __setitem__(self, key: np.ndarray, value: np.ndarray):
    self.update(key, value)

  def update(self, key: np.ndarray, value: np.ndarray):
    """Update elements' values"""
    # formalize indices
    if isinstance(key, (int, slice)):
      key = np.asarray(range(self._size)[key], dtype=np.int64)
    else:
      key = np.asarray(key, dtype=np.int64)
    key = key % self._size + self._base
    key = key.flatten()
    value = np.asarray(value, dtype=np.float64).flatten()
    # set values
    self._value[key] = value
    # update tree (all keys have the same depth)
    while key[0] > 1:
      self._value[key>>1] = self._value[key] + self._value[key^1]
      key >>= 1

  def sum(
    self,
    start: Optional[int] = None,
    end: Optional[int] = None
  ) -> np.ndarray:
    """Compute the sum of the given range [start, end)

    Args:
      start (Optional[int], optional): start index (included).
        Defaults to None.
      end (Optional[int], optional): end index (excluded).
        Defaults to None.

    Returns:
      np.ndarray: summation from [start, end)
    """
    '''Compute the sum of the given range [start, end)'''
    if (start == None) and (end == None):
      # shortcut
      return self._value[1]
    start, end, _ = slice(start, end).indices(self._size)
    start += self._base
    end += self._base
    res = 0.0
    while start < end:
      if start & 1:
        res += self._value[start]
      if end & 1:
        res += self._value[end-1]
      start = (start+1) >> 1
      end = end >> 1
    return res

  def index(self, value: np.ndarray) -> Union[int, np.ndarray]:
    """Return the largest index such that
    value[0:index+1].sum() < value
    """
    assert np.min(value) >= 0.0
    assert np.max(value) < self._value[1]
    # if input is a scalar, return should be a scalar too.
    one_value = np.isscalar(value)
    # convert to 1D array
    value = np.asarray(value, dtype=np.float64)
    orig_shape = value.shape
    value = value.flatten()
    inds = np.ones_like(value, dtype=np.int64)
    # find inds (all inds have the same depth)
    while inds[0] < self._base:
      inds <<= 1
      lsum = self._value[inds]
      d = lsum < value
      value -= lsum * d
      inds += d
    inds -= self._base
    inds = inds.reshape(orig_shape)
    return inds.item() if one_value else inds

# === Replay buffer ===

class RelativeIndex():
  def __init__(
    self,
    buffer: 'BaseBuffer',
    offsets: tuple = None,
    max_sizes: tuple = None
  ):
    """This class is used to relative indexing

    For example, if the buffer has 10 slots and the last element
    (pos) is at 6th slot, you can get the last element by absolute 
    indexing
    >>> buffer[6]
    by relative indexing (relative to `buffer.head`)
    >>> buffer.rel[-1]
    `-1` is the last element in the buffer.
    This is useful when you want to calculate N-step returns

    This also support vectorized relative indexing, for example:
    >>> rel = RelativeIndex(buffer, [1, 3, 5])
    >>> rel[-1]
    this is equivelent to
    >>> buffer[[0, 2, 4]]

    Note that relative indexing only supports int, np.ndarray and 
    slice. It does not support Ellipsis.

    Args:
      buffer (BaseBuffer): Buffer
      offsets (int, tuple, np.ndarray): The indices relative to.
        If None, it is set to `buffer.head`. Defaults to None.
      max_sizes
    """
    # Avoid circular import
    from rlchemy.lib.data.buffers import BaseBuffer
    assert isinstance(buffer, BaseBuffer)
    self.buffer = buffer
    if offsets is None:
      offsets = buffer.head
    if max_sizes is None:
      max_sizes = buffer.len_slots()
    self.offsets = np.index_exp[offsets]
    self.max_sizes = np.index_exp[max_sizes]
    assert len(self.offsets) == len(self.max_sizes)

  def toabs(
    self,
    key: Union[int, slice, List[int], Tuple[int], np.ndarray],
    offset: Union[int, List[int], Tuple[int], np.ndarray],
    max_size: int
  ):
    """Convert to absolute indexing"""
    if key is Ellipsis:
      raise NotImplementedError("Relative indexing does not"
        "support Ellipsis")
    # formalize indices (to np.ndarray)
    if isinstance(key, (int, slice)):
      key = np.asarray(range(max_size)[key], dtype=np.int64)
    else:
      key = np.asarray(key, dtype=np.int64)
    if not np.isscalar(offset):
      # vectorized indexing (*key.shape, *offset.shape)
      key = key.reshape(key.shape + (1,)*offset.ndim)
    key = (key + offset) % max_size
    return key
  
  def cvtkeys(
    self,
    keys: Union[int, slice, List[int], Tuple[int], np.ndarray]
  ) -> Tuple[np.ndarray]:
    """Convert relative keys to absolute keys"""
    keys = np.index_exp[keys]
    min_len = min(len(self.offsets), len(keys))
    abs_keys = []
    for dim in range(min_len):
      abs_key = self.toabs(
        keys[dim],
        self.offsets[dim],
        self.max_sizes[dim]
      )
      abs_keys.append(abs_key)
    if len(keys) > min_len:
      abs_keys.extend(keys[min_len:])
    elif len(self.offsets) > min_len:
      abs_keys.extend(self.offsets[min_len:])
    return tuple(abs_keys)

  def __getitem__(
    self,
    keys: Union[int, slice, List[int], Tuple[int], np.ndarray]
  ) -> Any:
    return self.buffer[self.cvtkeys(keys)]

  def __setitem__(
    self,
    keys: Union[int, slice, List[int], Tuple[int], np.ndarray],
    values: Any
  ) -> Any:
    self.buffer[self.cvtkeys(keys)] = values