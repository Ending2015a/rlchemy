# --- built in ---
import abc
import math
from typing import (
  Any,
  Dict,
  Iterable,
  KeysView,
  Optional,
  Tuple,
  Union
)
import collections
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.data.utils import RelativeIndex

__all__ = [
  'BaseBuffer',
  'ReplayBuffer',
  'DynamicBuffer'
]

# === Typing ===
Indices2D = Tuple[np.ndarray, np.ndarray]
Indices1D = np.ndarray
Indices = Union[int, slice, Indices1D, Indices2D]

# === Buffers ===

class BaseBuffer(metaclass=abc.ABCMeta):
  """The base class of replay buffers.
  BaseBuffer stores samples in a 2D manner (slots, batch, ...), where `slots`
  can be seen as the timesteps of each samples, and `batch` is the
  number of samples each time user `add()` into the buffer. `...` is the data
  dimensions.

  Features:
  * You can think this like a circular array.
  * Support arbitrary structure of nested samples (tuple, dict).
  * Support relative indexing from any position of the buffer.
  * Support sampling sequences of data.

  Cons (TODO):
  * Does not support episodic sampling.
  * Does not support dynamic size of samples.
  """
  def __init__(self, size: int, batch: Optional[int]=None):
    """Create a buffer
    
    Args:
      size (int): Buffer size, capacity
      batch (int, optional): batch size of each replay sample, commonly
        referred to as number of envs in a vectorized env. This value
        can be set to None for delayed setup. Or this value will be set
        automatically at the first time when user adds a batch of 
        samples to this buffer. Defaults to None.
    """
    if not isinstance(size, int) or size <= 0:
      raise ValueError(f"`size` must be greater than 0, got {size}")
    self._min_size: int = size
    self._batch: Optional[int] = batch
    self._slots: Optional[int] = None
    self._size: Optional[int] = None
    self._pos: int = 0
    self._full: bool = False
    self._data: Optional[Any] = None
    self.reset()
    # calculate buffer spaces
    if batch is not None:
      self._calc_space(batch)

  @property
  def capacity(self) -> int:
    """Return the capacity of the buffer"""
    return self._size
  
  @property
  def slots(self) -> int:
    """Return the number of total slots"""
    return self._slots
  
  @property
  def batch(self) -> int:
    """Return batch size"""
    return self._batch
  
  @property
  def head(self) -> int:
    """Return the index of the first slot"""
    return self._pos if self.isfull else 0

  @property
  def tail(self) -> int:
    """Return the index of the last slot"""
    return self._pos
  
  @property
  def isnull(self) -> bool:
    """Return True, if the buffer space is not created"""
    return self._data is None

  @property
  def isfull(self) -> bool:
    """Return True, if the buffer is full"""
    return self._full

  @property
  def ready_for_sample(self) -> bool:
    """Return True, if it is ready for sampling"""
    return True

  @property
  def data(self) -> Optional[Any]:
    """Retrieve data"""
    return self._data

  @property
  def rel(self) -> RelativeIndex:
    """Relative indexing"""
    return RelativeIndex(self)

  def reset(self):
    """Reset buffer"""
    self._pos = 0
    self._full = False
    self._data = None

  def ravel_index(self, indices: Indices2D) -> Indices1D:
    """Ravel 2D indices to 1D indices"""
    assert not self.isnull
    return np.ravel_multi_index(indices, (self.slots, self.batch))

  def unravel_index(self, indices: Indices1D) -> Indices2D:
    """Unravel 1D indices to 2D indices"""
    assert not self.isnull
    return np.unravel_index(indices, (self.slots, self.batch))

  def add(self, data: Any) -> Indices2D:
    """Add new batch data into the buffer

    Args:
      data (Any): Any nested type of data. Note that the python list
        is treated as a single element. Each data must have shape 
        (b, *)

    Return:
      Indices2D: 2D indices of the samples
    """
    # count the batch dimension
    arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
    assert not np.isscalar(arr), f'rank must be > 0'
    n_samples = len(arr)
    # prepare indices and copy data into the buffer
    cur_pos = self._pos
    self._set_data(data, indices=cur_pos)
    # update cursor position (circular buffer)
    self._pos += 1
    if self._pos >= self._slots:
      self._full = True
      self._pos = self._pos % self._slots
    return (cur_pos, np.arange(n_samples))

  def update(self, data: Any, indices: Indices):
    """Update buffer data"""
    self._set_data(data, indices=indices)

  def len_slots(self) -> int:
    """Return the number of filled slots"""
    return (self._slots if self.isfull else self._pos)

  def __len__(self) -> int:
    """Return the number of total samples"""
    if self.isnull:
      return 0
    return (self._size if self.isfull else self._pos*self._batch)

  def __getitem__(self, key: Indices) -> Any:
    """Get a slice of data
    
    Args:
      key(Indices): Indices.

    Returns:
      Any: A slice of nested data
    """
    return self._get_data(key)

  def __setitem__(self, key: Indices, value: Any):
    """Set slice of data
    
    Args:
      key (Indices): Indices slices
      value (Any): New data
    """
    self._set_data(value, indices=key)

  def _calc_space(self, batch: int):
    """Calculate buffer spaces"""
    assert isinstance(batch, int)
    assert batch > 0
    self._batch = batch
    self._slots = math.ceil(self._min_size/batch)
    self._size = self._slots * self._batch # true size

  def _melloc(self, data: Any):
    """Create spaces from the given data example"""
    def _create_space(v):
      v = rl_utils.to_numpy(v)
      shape = (self._slots, self._batch)
      if len(v.shape) > 1:
        shape += v.shape[1:]
      return np.zeros(shape, dtype=v.dtype)
    # calculate buffer spaces
    if self._batch is None:
      arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
      assert not np.isscalar(arr), "rank must be > 0"
      self._calc_space(len(arr))
    self._data = rl_utils.map_nested(data, op=_create_space)

  def _get_data(self, indices: Indices) -> Any:
    assert not self.isnull, "Buffer space not created"
    _slice_op = lambda v: v[indices]
    return rl_utils.map_nested(self._data, _slice_op)

  def _set_data(self, data: Any, indices: Indices):
    assert indices is not None, "`indices` not set"
    # create space if the space is empty
    if self.isnull:
      self._melloc(data)
    # assign to buffer
    def _assign_op(data_tuple):
      new_data, data = data_tuple
      data[indices] = rl_utils.to_numpy(new_data).astype(data.dtype)
    rl_utils.map_nested_tuple((data, self._data), _assign_op)

@rl_registry.register.buffer(
  ['replay_buffer', 'replay'],
  default=True
)
class ReplayBuffer(BaseBuffer):
  """A replay buffer to store nested type of data (tuple, dict)
  Note that the python list is treated as a single element and
  will be converted to np.ndarray to store.

  Example:
  >>> buffer = ReplayBuffer(3, 1)
  >>> buffer.add(a=[0.1], b=([1], [True]))
  >>> buffer.add(a=[0.2], b=([2], [True]))
  >>> buffer.data
  {'a': array([[0.1, 0.2, 0.0]]),
    'b': (array([[1, 2, 0]]), array([[ True, True, False]]))}
  """
  def keys(self) -> KeysView:
    """Return keys"""
    return (
      collections.abc.KeysView([])
      if self.isnull
      else self._data.keys()
    )
  
  def __contains__(self, key: str) -> bool:
    """Return True, if key is in the dict"""
    return not self.isnull and key in self._data.keys()

  def add(self, **data) -> Indices2D:
    """Add one sample to the buffer"""
    return super().add(data)

  def update(self, indices: Indices, **data):
    """Update buffer contents"""
    super().update(data, indices=indices)

  def _assert_keys_exist(self, keys: Iterable[str]):
    for key in keys:
      assert key in self, f'Key "{key}" does not exist'

  def _set_data(self, data: Dict[str, Any], indices: Indices):
    if not self.isnull:
      self._assert_keys_exist(data.keys())
    super()._set_data(data, indices)

@rl_registry.register.buffer(
  ['dynamic_buffer', 'dynamic']
)
class DynamicBuffer(ReplayBuffer):
  def __init__(self, batch: int=None):
    """A dynamic replay buffer which has unlimited spaces to store 
    sequential nested data. This kind of buffer is used by on-policy 
    algo, e.g. PPO, which needs to reset buffer frequently and has an
    uncertained amount of samples. Note that `buffer.make()` must be 
    called before sampling data. Use this buffer as the following
    procedure:
    ```python
    buffer = DynamicBuffer()
    for epoch in range(total_epochs):
      # 1. Clear the buffer
      buffer.reset()

      # 2. Interact with env
      for step in range(total_steps):
        data = env.step()
        buffer.add(data)
      
      # 3. Prepare for training
      # this step will convert all the data into np.ndarray
      buffer.make()
      # you can also do additional calculations after making the buffer
      # e.g. generalized advantage estimation (GAE)
      buffer['gae'] = compute_gae(buffer)
      
      # 4. Create sampler and start training your agent
      sampler = Sampler(buffer)
      batch = sampler.sample()
      td_err, loss = model.train_step(batch)

      # 5. Update buffer contents if needed
      # e.g. you can update the sample priority
      # if you are using the prioritized replay sampler
      buffer.update(...)
      sampler.update(td_err)
    ```

    Args:
      batch (int, optional): batch size of replay samples, commonly 
        referred to as number of envs in a vectorized env. This 
        value is automatically set when user first adds a batch 
        of data. Defaults to None.
    """
    self._batch: Optional[int] = batch
    self._ready_for_sample: bool = False
    self.reset()

  @property
  def capacity(self) -> float:
    """Return the capacity of the buffer"""
    return float('inf')

  @property
  def slots(self) -> int:
    """Return number of slots"""
    return self._pos

  @property
  def head(self) -> int:
    """Return the index of the first slot"""
    return 0

  @property
  def ready_for_sample(self) -> bool:
    """Return True if buffer is ready for sampling"""
    return self._ready_for_sample

  def reset(self):
    """Reset buffer"""
    super().reset()
    self._ready_for_sample = False

  def add(self, **data) -> Indices2D:
    """Add new batch data into the buffer"""
    if self.ready_for_sample:
      raise RuntimeError("Buffer can\'t add data when it "
        "is ready for sampling")
    arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
    assert not np.isscalar(arr), "rank must be > 0"
    n_samples = len(arr)
    # copy data into the buffer
    cur_pos = self._pos
    self._append_data(data)
    # increase number of batch samples
    self._pos += 1
    return (cur_pos, np.arange(n_samples))

  def update(self, indices: Indices, **data):
    """Update buffer contents
    You should call `make()` before calling this function
    """
    if not self.ready_for_sample:
      raise RuntimeError("Call `buffer.make()` before calling "
        "`buffer.update()`")
    super().update(indices=indices, **data)

  def make(self):
    """Prepare for sampling
    Convert list to np.ndarray
    """
    if self.ready_for_sample:
      raise RuntimeError("The buffer has already made.")
    self._data = rl_utils.nested_to_numpy(self._data)
    self._ready_for_sample = True

  def len_slots(self) -> int:
    return self._pos

  def __len__(self) -> int:
    return 0 if self.isnull else self._pos*self._batch

  def _calc_space(self, batch: int):
    """Calculate buffer spaces"""
    assert isinstance(batch, int)
    assert batch > 0
    self._batch = batch

  def _melloc(self, data: Any):
    """Create buffer space"""
    _create_space = lambda v: []
    if self._batch is None:
      arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
      assert not np.isscalar(arr), "rank must be > 0"
      self._calc_space(len(arr))
    self._data = rl_utils.map_nested(data, op=_create_space)
  
  def _append_data(self, data: Any):
    if self.isnull:
      self._melloc(data)
    if self.ready_for_sample:
      raise RuntimeError("The buffer can not append data after "
        "calling `buffer.make()`.")
    self._assert_keys_exist(data.keys())
    def _append_op(data_tuple):
      new_data, data = data_tuple
      data.append(rl_utils.to_numpy(new_data))
    rl_utils.map_nested_tuple((data, self.data), _append_op)
