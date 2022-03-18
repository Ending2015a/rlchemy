# --- built in ---
import abc
from typing import (
  Any,
  Iterator,
  Optional,
  Tuple,
  Union
)
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.data.utils import RelativeIndex, SegmentTree
from rlchemy.lib.data.buffers import (
  BaseBuffer,
  ReplayBuffer,
  DynamicBuffer
)

__all__ = [
  'BaseSampler',
  'UniformSampler',
  'PrioritizedSampler',
  'PriorSampler', # alias: PrioritizedSampler
  'PermuteSampler'
]
# === typing ===
Indices2D = Tuple[np.ndarray, np.ndarray]
Indices1D = np.ndarray
Indices = Union[int, slice, Indices1D, Indices2D]

class BaseSampler(metaclass=abc.ABCMeta):
  def __init__(self, buffer: BaseBuffer):
    if not isinstance(buffer, BaseBuffer):
      raise ValueError("`buffer` must be an instance of BaseBuffer, "
        f"got {type(buffer)}")
    self._buffer = buffer
    self._cached_inds: Indices2D = None
  
  @property
  def buffer(self) -> BaseBuffer:
    return self._buffer

  @property
  def indices(self) -> Indices2D:
    """Cached indices"""
    return self._cached_inds

  @property
  def rel(self) -> RelativeIndex:
    """Relative indexing, relative to sampled indices"""
    return RelativeIndex(
      self._buffer,
      self._cached_inds,
      (self._buffer.len_slots(), self._buffer.batch)
    )

  def __call__(self, *args, **kwargs) -> Any:
    """A shortcut to self.sample()"""
    return self.sample(*args, **kwargs)

  @abc.abstractmethod
  def sample(self, batch_size, seq_len) -> Any:
    """Random sample replay buffer"""
    raise NotImplementedError

  def add(self, *args, **kwargs) -> Indices2D:
    """Add samples to the buffer"""
    return self.buffer.add(*args, **kwargs)

  def update(self, *args, **kwargs):
    """Update sampled data to the buffer"""
    assert self._cached_inds is not None
    self.buffer.update(*args, **kwargs, indices=self._cached_inds)

@rl_registry.register.sampler(
  ['uniform_sampler', 'uniform'],
  default = True
)
class UniformSampler(BaseSampler):
  def __init__(self, buffer: BaseBuffer):
    """Uniform sampler samples buffer uniformly.
    This sampler is mainly for off-policy algorithms.

    Args:
      buffer (BaseBuffer): Replay buffer
    """
    super().__init__(buffer)

  def sample(
    self,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    fixed_seq: bool = False
  ) -> Any:
    """Randomly sample a batch or a batch sequence of data from the 
    replay buffer

    Args:
      batch_size (int, optional): Batch size to sample. Defaults to None.
      seq_len (int, optional): Length of sequences. Defaults to None.
      fixed_seq (bool, optional): Sample sequential data from the fixed
        positions. For example, if you sample sequences with length 4,
        the sampler will sample the sequences from 0, 4, 8, 12, ....
        Otherwise, it will sample from arbitrary positions. This feature
        is essential for RNN type networks, which require hidden states
        for training. Fixing the sampling position is more stable.
        Defaults to True. TODO: unittest
    
    Returns:
      Any: Returns a slice of data from the replay buffer. Each element has 
        shape (b, *data.shape) if seq_len is None, otherwise, 
        (b, seq, *data.shape).
    """
    buf = self.buffer
    if not buf.ready_for_sample:
      raise RuntimeError("Buffer is not ready for sampling, "
        "call `buffer.make()` before sampling")
    if batch_size is None:
      batch_size = len(buf)
    if seq_len is None:
      # uniformly sample buffer positions
      inds = np.random.randint(len(buf), size=batch_size)
      inds1, inds2 = buf.unravel_index(inds)
    else:
      if fixed_seq:
        # sample slots indices/seq_len
        inds1, inds2 = self._sample_fixed_seq(batch_size, seq_len)
      else:
        # sampling sequences from arbitrary positions
        inds1, inds2 = self._sample_random_seq(batch_size, seq_len)
    self._cached_inds = (inds1, inds2)
    return self.buffer[self._cached_inds]

  def _sample_fixed_seq(
    self,
    batch_size: int,
    seq_len: int,
  ) -> Indices2D:
    """Sample sequences from fixed positions"""
    buf = self.buffer
    inds1_dv_seq_len = np.random.randint(
      buf.len_slots()//seq_len - 1,
      size = (batch_size, 1)
    )
    # sample batch indices
    inds2 = np.random.randint(
      buf.batch,
      size = (batch_size, 1)
    )
    # shifting the indices to avoid sampling
    # across the end of the buffer
    head_dv_seq_len = (buf.head + seq_len-1)//seq_len
    len_slots_dv_seq_len = buf.len_slots()//seq_len
    inds1_dv_seq_len += head_dv_seq_len
    inds1_dv_seq_len %= len_slots_dv_seq_len
    inds1 = inds1_dv_seq_len * seq_len
    # broadcast shapes and indices
    inds1 = (inds1 + np.arange(seq_len)) % buf.len_slots()
    inds2 = np.broadcast_to(inds2, inds1.shape)
    return inds1, inds2

  def _sample_random_seq(
    self,
    batch_size: int,
    seq_len: int
  ) -> Indices2D:
    """Sample sequences from arbitrary positions"""
    buf = self.buffer
    inds = np.random.randint(
      len(buf) - seq_len*buf.batch,
      size = (batch_size, 1)
    )
    inds1, inds2 = buf.unravel_index(inds)
    # shifting the indices to avoid sampling
    # across the end of the buffer
    inds1 = inds1 + buf.head
    # broadcast shapes and indices
    inds1 = (inds1 + np.arange(seq_len)) % buf.len_slots()
    inds2 = np.broadcast_to(inds2, inds1.shape)
    return inds1, inds2

@rl_registry.register.sampler(
  ['prioritized_sampler', 'prior_sampler',
   'prior', 'per']
)
class PrioritizedSampler(BaseSampler):
  def __init__(
    self,
    buffer: ReplayBuffer,
    alpha: float,
    weight_key: str = 'w'
  ):
    """An implementation of Prioritized Experience Replay
    See
    * arXiv:1511.05952, "Prioritized Experience Replay"

    Args:
      buffer (ReplayBuffer): Replay buffer.
      alpha (float): Prioritization exponent.
      weight_key (str, optional): keys to store weights. One can access
        weights from the sampled batches with this key. Defaults to 'w'.
    """
    super().__init__(buffer)
    if not isinstance(buffer, BaseBuffer):
      raise ValueError(f"{type(self)} does not support {type(buffer)}")
    if isinstance(buffer, DynamicBuffer):
      raise ValueError(f"{type(self)} does not support DynamicBuffer")
    self._weight_tree = None
    self._weight_key = weight_key
    self._alpha = alpha
    self._max_w = 1.0
    self._min_w = 1.0
    self._eps = np.finfo(np.float32).eps.item()

  def add(self, *args, **kwargs) -> Indices2D:
    """Add samples to the buffer and weight tree"""
    _indices = self.buffer.add(*args, **kwargs)
    indices = self.buffer.ravel_index(_indices)
    if self._weight_tree is None:
      self._melloc()
    self._weight_tree[indices] = self._max_w**self._alpha
    return _indices

  def update(self, *args, **kwargs):
    """Update sampled data to the buffer"""
    w = kwargs.pop(self._weight_key, None)
    if w is not None:
      # update tree weights
      w = np.abs(rl_utils.to_numpy(w)) + self._eps
      flat_w = w.flatten()
      inds = self.buffer.ravel_index(self._cached_inds)
      self._weight_tree[inds] = flat_w ** self._alpha
      self._max_w = max(self._max_w, np.max(w))
      self._min_w = min(self._min_w, np.min(w))
    self.buffer.update(*args, **kwargs, indices=self._cached_inds)

  def sample(
    self,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    fixed_seq: bool = False,
    beta: float = 0.0
  ) -> Any:
    """Randomly sample a batch or a batch sequences of data from 
    the replay buffer

    NOTE that we dont recommand you to sample sequences with this
    sampler, since it may sample a sequences that exceeds the tail
    of the circular buffer.

    Args:
      batch_size (int, optional): Batch size to sample. Defaults to None.
      seq_len (int, optional): Length of sequences. Defaults to None.
      fixed_seq (bool, optional): NOTE not implemented.
      beta (float, optional): Importance sampling exponent. Defaults to 0.0.

    Returns:
      Any: Returns a slice of data from the replay buffer. Each 
        element has shape (b, *data.shape) if seq_len is None, 
        otherwise, (b, seq, *data.shape).
    """
    buf = self.buffer
    if not buf.ready_for_sample:
      raise RuntimeError("Buffer is not ready for sampling, "
        "call `buffer.make()` before sampling")
    if batch_size is None:
      batch_size = len(buf)
    if seq_len is None:
      samp = np.random.rand(batch_size) * self._weight_tree.sum()
    else:
      bound = self._weight_tree.sum()
      samp = np.random.rand(batch_size).reshape(-1, 1) * bound
    inds = self._weight_tree.index(samp)
    inds1, inds2 = buf.unravel_index(inds)
    if seq_len is not None:
      inds1 = inds1 + np.arange(seq_len)
      inds2 = np.broadcast_to(inds2, inds1.shape)
    inds1 = inds1 % buf.len_slots()
    self._cached_inds = (inds1, inds2)
    inds = buf.ravel_index(self._cached_inds)
    # calculate sample weights
    weight = (self._weight_tree[inds] / self._min_w) ** (-beta)
    weight = weight / np.max(weight)
    weight = weight.reshape(inds1.shape)
    # get batch data
    batch = self.buffer[self._cached_inds]
    batch[self._weight_key] = weight
    return batch

  # --- private methods ---
  def _melloc(self):
    """Create segment tree space"""
    self._weight_tree = SegmentTree(self.buffer.capacity)

# alias
PriorSampler = PrioritizedSampler

@rl_registry.register.sampler(
  ['permute_sampler', 'permute']
)
class PermuteSampler(BaseSampler):
  def __init__(self, buffer: BaseBuffer):
    """Permute sampler samples buffer by random permutation. It
    returns an iterator iterates through all rollout data.

    Args:
      buffer (BaseBuffer): Replay buffer.
    """
    super().__init__(buffer)
  
  def sample(
    self,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    fixed_seq: bool = False
  ) -> Iterator:
    """Create an iterator which randomly iterate through all data 
    in the replay buffer with a given batch size and sequence length.

    Args:
      batch_size (int, optional): Batch size to sample. Defaults to None.
      seq_len (int, optional): Length of sequences. If it's None, the sampled
        data has shape (b, *data.shape), otherwise, (b, seq, *data.shape).
        Defaults to None.
      fixed_seq(bool, optional): Sample sequential data from the fixed
        positions. For example, if you sample sequences with length 4,
        the sampler will sample the sequences from 0, 4, 8, 12, ....
        Otherwise, it will sample from arbitrary positions. This feature
        is essential for RNN type networks, which require hidden states
        for training. Fixing the sampling position is more stable.
        Defaults to True. TODO: unittest
    
    Returns:
      Iterator: Returns an iterator which produces slices of data from the replay 
        buffer. Each element has shape (b, *data.shape) if seq_len is None, 
        otherwise, (b, seq, *data.shape).
    """
    buf = self.buffer
    if not buf.ready_for_sample:
      raise RuntimeError("Buffer is not ready for sampling, "
        "call `buffer.make()` before sampling")
    return self._iter(
      batch_size = batch_size,
      seq_len = seq_len,
      fixed_seq = fixed_seq
    )

  def _iter(
    self,
    batch_size: Optional[int] = None,
    seq_len: Optional[int] = None,
    fixed_seq: bool = False
  ) -> Iterator:
    """Create an iterator which iterates through the whole buffer"""
    buf = self.buffer
    if batch_size is None:
      batch_size = len(buf)
    if seq_len is None:
      inds = np.arange(len(buf))
      inds1, inds2 = buf.unravel_index(inds)
    else:
      if fixed_seq:
        inds1, inds2 = self._sample_fixed_seq(seq_len)
      else:
        inds1, inds2 = self._sample_random_seq(seq_len)
    # shuffle the indices of the indices of the samples
    permute = np.arange(len(inds1))
    np.random.shuffle(permute)
    start_ind = 0
    while start_ind < len(inds1):
      slice_ = range(start_ind, start_ind+batch_size)
      indices = np.take(permute, slice_, mode='wrap')
      self._cached_inds = (inds1[indices], inds2[indices])
      # return iterator
      yield self.buffer[self._cached_inds]
      start_ind += batch_size

  def _sample_fixed_seq(
    self,
    seq_len: int
  ) -> Indices2D:
    """Sample sequences from fixed positions"""
    buf = self.buffer
    h_dv_seq_len = buf.len_slots()//seq_len - 1
    w = buf.batch
    # generate grid indices
    inds1_dv_seq_len = np.repeat(np.arange(h_dv_seq_len), w)
    inds2 = np.tile(np.arange(w), h_dv_seq_len)
    # reshape
    inds1_dv_seq_len = inds1_dv_seq_len.reshape(-1, 1)
    inds2 = inds2.reshape(-1, 1)
    # shifting the indices to avoid sampling
    # across the end of the buffer
    head_dv_seq_len = (buf.head + seq_len-1)//seq_len
    len_slots_dv_seq_len = buf.len_slots()//seq_len
    inds1_dv_seq_len += head_dv_seq_len
    inds1_dv_seq_len %= len_slots_dv_seq_len
    inds1 = inds1_dv_seq_len * seq_len
    # broadcast shapes and indices
    inds1 = (inds1 + np.arange(seq_len)) % buf.len_slots()
    inds2 = np.broadcast_to(inds2, inds1.shape)
    return inds1, inds2

  def _sample_random_seq(
    self,
    seq_len: int
  ) -> Indices2D:
    """Sample sequences from arbitrary positions"""
    buf = self.buffer
    inds = np.arange(len(buf)-seq_len*buf.batch).reshape(-1, 1)
    inds1, inds2 = buf.unravel_index(inds)
    # shifting the indices to avoid sampling across the end of
    # the buffer
    inds1 = inds1 + buf.head
    # broadcast shapes and indices
    inds1 = (inds1 + np.arange(seq_len)) % buf.len_slots()
    inds2 = np.broadcast_to(inds2, inds1.shape)
    return inds1, inds2
