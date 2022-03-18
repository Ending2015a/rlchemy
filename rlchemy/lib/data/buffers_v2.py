# --- built in ---
import abc
import math
from typing import (
  Any,
  Dict,
  Iterable,
  List,
  Optional,
  Tuple,
  Union
)
import collections
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib.data.utils import RelativeIndex
from rlchemy.lib.registry import registry as rl_registry

__all__ [
  'TimeSlots',
  'DynamicTimeSlots'
]

Indices2D = Tuple[np.ndarray, np.ndarray]
Indices1D = np.ndarray
Indices = Union[int, slice, Indices1D, Indices2D]

class TimeSlots(metaclass=abc.ABCMeta):
  def __init__(
    self,
    min_capacity: int,
    num_entries: Optional[int] = None
  ):
    """Circular buffer to store sequential data.
    The data is stored in a 2D structure (slots, batch, ...), where `slots`
    indicates the timesteps of each data sample, and `batch` is the number
    of samples each time we `add()` to the buffer. Commonly it refers to
    the number parallel environments in a vectorized env. `...` is the
    data dimensions. `size` is the minimum capacity of the buffer. The real
    size would be 
    ```
    real_size = ceil(size / batch) * batch
    ```
    It is recommanded to set a number that can be devided by `batch`.

    Args:
      min_capacity (int): minimum capacity of the buffer.
      num_entries (int, optional): batch size of each data, commonly referred to
        as the number of environments in a vectorized env. This value would
        be set automatically at the first time when user adds a batch of data
        into the buffer. Defaults to None.
    """
    if not isinstance(min_capacity, int) or min_capacity <= 0:
      raise ValueError(f"`size` must be greater than 0, got {min_capacity}")
    self._min_size: int = min_capacity
    self._batch: Optional[int] = num_entries
    self._slots: Optional[int] = None
    self._size: Optional[int] = None
    self._pos: int = 0
    self._full: bool = False
    self._data: Optional[Any] = None
    self.reset()
    # calculate buffer spaces
    if num_entries is not None:
      self._calc_space(num_entries)

  @property
  def capacity(self) -> Optional[int]:
    """Return the capacity of the buffer
    Note that this value is `None` before the space being created
    """
    return self._size

  @property
  def slots(self) -> Optional[int]:
    """Return the number of total slots
    Note that this value is `None` before the space being created
    """
    return self._slots

  @property
  def batch(self) -> Optional[int]:
    """Return the number of batch
    Note that this value is `None` before the space being created
    """
    return self._batch

  @property
  def head(self) -> int:
    """Return the index of head slot of the circular buffer"""
    return self._pos if self.isfull else 0
  
  @property
  def tail(self) -> int:
    """Return the index of tail slot of the circular buffer"""
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
    """Return True, if the buffer is ready for sampling"""
    return True
  
  @property
  def data(self) -> Optional[Any]:
    """Retrieve data"""
    return self._data

  @property
  def rel(self) -> RelativeIndex:
    """Relatve indexing"""
    return RelativeIndex(self)
  
  def reset(self):
    """Reset buffer"""
    self._pos = 0
    self._full = False
    self._data = None

  def ravel_index(self, indices: Indices2D) -> Indices1D:
    """Ravel 2D indices to 1D indices
    Note that TimeSlots stores data in a 2D structure (slots, batch, ...)
    So the 2D indices are the indices of `slots` and `batch`. The raveled
    1D indices are the indices of `slots * batch`
    """
    assert not self.isnull
    return np.ravel_multi_index(indices, (self.slots, self.batch))
  
  def unravel_index(self, indices: Indices1D) -> Indices2D:
    """Unravel 1D indices to 2D indices
    See `ravel_index()`
    """
    assert not self.isnull
    return np.unravel_index(indices, (self.slots, self.batch))

  def add_batch(self, data: Any) -> Indices2D:
    """Add new batch data into the buffer

    Args:
      data (Any): Any nested type of data. Note that the python list
      is considered as a single element and will be converted to np.ndarray.
      Each data must be shape (batch, ...)

    Returns:
      Indices2D: 2D indices of the samples
    """
    # count the leading dimension (batch)
    arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
    assert not np.isscalar(arr), f"rank must be > 0"
    n_samples = len(arr)
    # copy data into the buffer
    cur_pos = self._pos
    self._set_data(data, indices=self._pos)
    # move cursor position (circular)
    self._pos += 1
    if self._pos >= self._slots:
      self._full = True
      self._pos = self._pos % self._slots
    indices = (cur_pos, np.arange(n_samples))
    return indices

  def update(self, data: Any, indices: Indices):
    """Update buffer data"""
    self._set_data(data, indices=indices)

  def len_slots(self) -> int:
    """Return the number of filled slots"""
    return (self._slots if self.isfull else self._pos)

  def __len__(self) -> int:
    """Return the number of total samples (batch * len_slots)"""
    if self.isnull:
      return 0
    if self.isfull:
      return self._size
    else:
      return self._pos * max(1, self._batch)
  
  def __getitem__(self, key: Indices) -> Any:
    """Get a slice of data

    Args:
      key (Indices): indices

    Returns:
      Any: slice of nested data
    """
    return self._get_data(key)

  def __setitem__(self, key: Indices, value: Any):
    """Set a slice of data

    Args:
      key (Indices): indices
      value (Any): new data
    """
    self._set_data(value, indices=key)

  def _calc_space(self, batch: int):
    """Calculate buffer space"""
    assert isinstance(batch, int)
    assert batch > 0
    self._batch = batch
    self._slots = math.ceil(self._min_size/batch)
    self._size = self._slots * self._batch

  def _melloc(self, data: Any):
    """Create spaces from the given data sample"""
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

  def _set_data(
    self,
    data: Any,
    indices: Indices
  ):
    assert indices is not None, "`indices` not set"
    # create space if the space is empty
    if self.isnull:
      self._melloc(data)
    # assign to buffer
    def _assign_op(data_tuple):
      new_data, data = data_tuple
      data[indices] = rl_utils.to_numpy(new_data).astype(data.dtype)
    rl_utils.map_nested_tuple((data, self._data), _assign_op)

class DynamicTimeSlots(TimeSlots):
  def __init__(self, num_entries: Optional[int]=None):
    """A dynamic replay buffer which has unlimited spaces to store 
    sequential nested data. This kind of buffer is used by on-policy 
    algo, e.g. PPO, which needs to reset buffer frequently and has an
    uncertained amount of samples. Note that `buffer.make()` must be 
    called before sampling data.
    This buffer also support storing non-batching trajectory by setting
    `num_entries` to 0.

    Args:
      num_entries (int, optional): batch size of replay samples, commonly 
        referred to as number of envs in a vectorized env. This 
        value is automatically set when user first adds a batch 
        of data. Defaults to None.
    """
    self._batch: Optional[int] = num_entries
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
    return self._pos

  @property
  def ready_for_sample(self) -> bool:
    """Return True if the buffer is ready for sampling"""
    return self._ready_for_sample
  
  def reset(self):
    """Reset buffer"""
    super().reset()
    self._ready_for_sample = False
  
  def add_batch(self, **data) -> Indices2D:
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
    # update cursor position
    self._pos += 1
    return (cur_pos, np.arange(n_samples))

  def add(self, **data) -> Indices1D:
    """Add new data into the buffer (non-batching)"""
    if self.ready_for_sample:
      raise RuntimeError("Buffer can\'t add data when it "
        "is ready for sampling")
    cur_pos = self._pos
    self._append_data(data)
    # update cursor position
    self._pos += 1
    return (cur_pos, )

  def update(self, indices: Indices, **data):
    """Update buffer contents
    You should call `make()` before calling this function
    """
    if not self.ready_for_sample:
      raise RuntimeError('Call `buffer.make()` before calling '
        '`buffer.update()`')
    super().update(indices=indices, **data)

  def make(self):
    """Prepare for sampling
    Convert list to np.ndarray
    """
    if self.ready_for_sample:
      raise RuntimeError('The buffer has already made.')
    self._data = rl_utils.nested_to_numpy(self._data)
    self._ready_for_sample = True

  def len_slots(self) -> int:
    return self._pos

  def __len__(self) -> int:
    if self.isnull:
      return 0
    return self._pos * max(1, self._batch)
  
  def _calc_space(self, batch: int):
    """Calculate buffer spaces"""
    assert isinstance(batch, int)
    self._batch = batch

  def _melloc(self, data: Any):
    """Create buffer space"""
    _create_space = lambda v: []
    if self._batch is None:
      arr = rl_utils.to_numpy(next(iter(rl_utils.iter_nested(data))))
      assert not np.isscalar(arr), "rank must > 0"
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

@rl_registry.registry_class('buffer')
class BaseBuffer(metaclass=abc.ABCMeta):
  def __init__(self, min_capacity: int, batch: Optional[int]=None):
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

    Cons:
    * Does not support episodic sampling.
    * Does not support dynamic size of samples.

    Args:
      min_capacity (int): buffer size, capacity
      batch (int, optional): batch size of each replay sample, commonly
        referred to as number of envs in a vectorized env. This value
        can be set to None for delayed setup. Or this value will be set
        automatically at the first time when user adds a batch of 
        samples to this buffer. Defaults to None.
    """
    self._timeslots = TimeSlots(
      min_capacity = min_capacity,
      num_entris = batch
    )
  
  @property
  def data(self) -> Any:
    return self._timeslots.data
  
  def add(self, data: Any) -> Indices2D:
    return self._timeslots.add_batch(data)



# class Trajectory():
#   def __init__(self):
#     self._pos: int = 0
#     self._full: bool = False
#     self._data: Optional[Any] = None
#     self.reset()

#   @property
#   def capacity(self) -> float:
#     return float('inf')
  
#   @property
#   def slots(self) -> int:
#     return self._pos

#   @property
#   def head(self) -> int:
#     return 0
  
#   @property
#   def ready_for_sample(self) -> bool:
#     return self._ready_for_sample
  
#   def reset(self):
#     self._pos = 0
#     self._full = False
#     self._data = None
#     self._ready_for_sample = False
  
#   def add(self, **data) -> Indices1D:
#     """Add new batch data into the buffer"""
#     if self.ready_for_sample:
#       raise RuntimeError("Buffer can\'t add data when it "
#         "is ready for sampling")
#     # copy data into the buffer
#     cur_pos = self._pos
#     self._append_data(data)
#     # increase number of batch samples
#     self._pos += 1
#     return (cur_pos, )
  
#   def update(self, indices: Indices, **data):
#     """Update buffer contents
#     You should call `make()` before calling this function
#     """
#     if not self.ready_for_sample:
#       raise RuntimeError('Call `buffer.make()` before calling '
#         '`buffer.update()`')
#     self._set_data(data, indices=indices)

#   def make(self):
#     """Prepare for sampling
#     Convert list to np.ndarray
#     """
#     if self.ready_for_sample:
#       raise RuntimeError("The buffer has already made.")
#     self._data = rl_utils.nested_to_numpy(self._data)
#     self._ready_for_sample = True

#   def len_slots(self) -> int:
#     return self._pos

#   def __len__(self) -> int:
#     return 0 if self.isnull else self._pos

#   def __getitem__(self, key: Indices) -> Any:
#     return self._get_data(key)
  
#   def __setitem__(self, key: Indices, value: Any):
#     self._set_data(value, indices=key)

#   def _melloc(self, data: Any):
#     _create_space = lambda v: []
#     self._data = rl_utils.map_nested(data, op=_create_space)
  
#   def _get_data(self, indices: Indices) -> Any:
#     _slice_op = lambda v: v[indices]
#     return rl_utils.map_nested(self._data, _slice_op)
  
#   def _set_data(
#     self,
#     data: Any,
#     indices: Indices
#   ):
#     assert indices is not None, "`indices` is not set"
#     # create space if the space is empty
#     if self.isnull:
#       self._melloc(data)
#     # assign to buffer
#     def _assign_op(data_tuple):
#       new_data, data = data_tuple
#       data[indices] = rl_utils.to_numpy(new_data).astype(data.dtype)
#     rl_utils.map_nested_tuple((data, self._data), _assign_op)

#   def _append_data(self, data: Any):
#     if self.isnull:
#       self._melloc(data)
#     if self.ready_for_sample:
#       raise RuntimeError("The buffer can not append data after "
#         "calling `buffer.make()`.")
#     self._assert_keys_exist(data.keys())
#     def _append_op(data_tuple):
#       new_data, data = data_tuple
#       data.append(rl_utils.to_numpy(new_data))
#     rl_utils.map_nested_tuple((data, self.data), _append_op)

# class EpisodicBuffer():
#   def __init__(self, max_episodes: int, done_key='done'):
#     self._done_key = done_key
#     self._max_episodes = max_episodes
#     self._episodes_completed = []
#     self._episodes_not_completed = []

#   def reset(self):
#     self._episodes_completed = []
#     self._episodes_not_completed = []

#   def add(self, **data) -> Tuple[np.ndarray]:
#     self._append_data(data)

#   def _melloc(self, data: Any):
#     """Create buffer space"""
#     _create_space

#   def _append_data(self, data: Any):
#     def _append_op(data_)
  
#   rl_utils.map_nested_tuple((data, self.data))


# class EpisodicUnionSampler():