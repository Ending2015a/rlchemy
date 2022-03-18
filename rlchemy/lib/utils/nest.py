# --- built in ---
import itertools
from typing import (
  Any,
  Callable,
  Iterator,
  List,
  Optional,
  Tuple
)
# --- 3rd party ---
import gym
import numpy as np
# --- my module ---

__all__ = [
  'iter_nested',
  'map_nested',
  'iter_nested_space',
  'map_nested_space',
  'iter_nested_tuple',
  'map_nested_tuple',
  'nested_to_numpy',
  'nested_to_tensor',
  'unpack_structure',
  'pack_sequence',
  'flatten_space',
  'flatten'
]

def iter_nested(data: Any, sortkey: bool=False) -> Iterator[Any]:
  """Iterate over nested data structure
  Note: Use `tuple` instead of `list`. A list type
  object is treated as an item.

  For example:
  >>> data = {'a': (1, 2), 'b': 3}
  >>> list(v for v in iter_nested(data))
  [1, 2, 3]

  Args:
    data (Any): A nested data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  def _inner_iter_nested(data):
    if isinstance(data, dict):
      keys = sorted(data.keys()) if sortkey else data.keys()
      for k in keys:
        yield from _inner_iter_nested(data[k])
    elif isinstance(data, tuple):
      for v in data:
        yield from _inner_iter_nested(v)
    else:
      yield data
  return _inner_iter_nested(data)

def map_nested(
  data: Any,
  op: Callable,
  *args,
  sortkey: bool = False,
  **kwargs
) -> Any:
  """A nested version of map function
  NOTE: Use `tuple` instead of `list`. A list type 
  object is treated as an item.

  Args:
    data (Any): A nested data
    op (Callable): A function operate on each data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not callable(op):
    raise ValueError('`op` must be a callable')

  def _inner_map_nested(data):
    if isinstance(data, dict):
      keys = sorted(data.keys()) if sortkey else data.keys()
      return {k: _inner_map_nested(data[k])
            for k in keys}
    elif isinstance(data, tuple):
      return tuple(_inner_map_nested(v)
              for v in data)
    else:
      return op(data, *args, **kwargs)
  return _inner_map_nested(data)

def iter_nested_space(
  space: gym.spaces.Space,
  sortkey: bool = False
) -> Iterator[gym.spaces.Space]:
  """Iterate over nested gym space. Similar to iter_nested
  but it's for gym spaces.

  Args:
    space (gym.Space): Nested or non-nested gym space
    sortkey (bool): (deprecated) Whether to sort dict's key. 
      Defaults to False.
  """
  def _inner_iter_nested(space):
    if isinstance(space, (gym.spaces.Dict, dict)):
      for k in space:
        yield from _inner_iter_nested(space[k])
    elif isinstance(space, (gym.spaces.Tuple, tuple)):
      for v in range(len(space)):
        yield from _inner_iter_nested(space[v])
    else:
      yield space
  return _inner_iter_nested(space)

def map_nested_space(
  space: gym.spaces.Space,
  op: Callable,
  *args,
  sortkey: bool = False,
  **kwargs
) -> Any:
  """A nested version of map function. Similar to map_nested
  but it's for gym spaces.

  Args:
    space (gym.Space): Nested or non-nested gym space
    op (Callable): A function operate on each data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not callable(op):
    raise ValueError('`op` must be a callable')

  def _inner_map_nested(space):
    if isinstance(space, (gym.spaces.Dict, dict)):
      return {k: _inner_map_nested(space[k])
            for k in space}
    elif isinstance(space, (gym.spaces.Tuple, tuple)):
      return tuple(_inner_map_nested(space[v])
            for v in range(len(space)))
    else:
      return op(space, *args, **kwargs)
  return _inner_map_nested(space)

def iter_nested_tuple(
  data_tuple: Tuple[Any],
  sortkey: bool = False
) -> Iterator[Tuple[Any]]:
  """Iterate over a tuple of nested structures. Similar to iter_nested
  but it iterates each of each nested data in the input tuple.
  For example:
  >>> a = {'x': 1, 'y': (2, 3)}
  >>> b = {'u': 4, 'v': (5, 6)}
  >>> list(iter_nested_tuple((a, b)))
  [(1, 4), (2, 5), (3, 6)]

  Args:
    data_tuple (Tuple[Any]): A tuple of nested data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not isinstance(data_tuple, tuple):
    raise TypeError('`data_tuple` only accepts tuple, '
      f'got {type(data_tuple)}')
  def _inner_iter_nested(data_tuple):
    if isinstance(data_tuple[0], dict):
      keys = data_tuple[0].keys()
      keys = sorted(keys) if sortkey else keys
      for k in keys:
        yield from _inner_iter_nested(
            tuple(data[k] for data in data_tuple))
    elif isinstance(data_tuple[0], tuple):
      for k in range(len(data_tuple[0])):
        yield from _inner_iter_nested(
          tuple(data[k] for data in data_tuple))
    else:
      yield data_tuple
  return _inner_iter_nested(data_tuple)

def map_nested_tuple(
  data_tuple: Tuple[Any],
  op: Callable,
  *args,
  sortkey: bool = False,
  **kwargs
) -> Tuple[Any]:
  """A nested version of map function. Similar to map_nested
  but it iterates each of each nested data in the input tuple.

  Args:
    data_tuple (Tuple[Any]): A tuple of nested data.
    op (Callable): A function operate on each data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  if not callable(op):
    raise ValueError('`op` must be a callable')
  if not isinstance(data_tuple, tuple):
    raise TypeError('`data_tuple` only accepts tuple, '
        f'got {type(data_tuple)}')
  def _inner_map_nested(data_tuple):
    if isinstance(data_tuple[0], dict):
      keys = data_tuple[0].keys()
      keys = sorted(keys) if sortkey else keys
      return {k: _inner_map_nested(
            tuple(data[k] for data in data_tuple))
          for k in keys}
    elif isinstance(data_tuple[0], tuple):
      return tuple(_inner_map_nested(
            tuple(data[idx] for data in data_tuple))
          for idx in range(len(data_tuple[0])))
    else:
      return op(data_tuple, *args, **kwargs)
  return _inner_map_nested(data_tuple)

def nested_to_numpy(
  data: Any,
  dtype: Optional[np.dtype] = None,
  sortkey: bool = False
) -> Any:
  """Convert all items in a nested data into 
  numpy arrays

  Args:
    data (Any): A nested data
    dtype (np.dtype): data type. Defaults to None.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.

  Returns:
    Any: A nested data same as `data`
  """
  # Avoid circular import
  from rlchemy.lib.utils import th_common
  op = lambda arr: th_common.to_numpy(arr, dtype=dtype)
  return map_nested(data, op, sortkey=sortkey)

def nested_to_tensor(
  data: Any,
  device: Optional['torch.device'] = None,
  dtype: Optional['torch.dtype'] = None,
  sortkey: bool = False
) -> Any:
  """Convert all items in a nested data into
  torch.Tensor

  Args:
    data (Any): a nested data
    device (torch.device, optional): torch device. Defaults to None.
    dtype (torch.dtype): torch data type. Defaults to None.
    sortkey (bool, optional): whether to sort dict's key.
      Defaults to False.
  """
  # Avoid circular import
  from rlchemy.lib.utils import th_common
  op = lambda arr: th_common.to_tensor(arr, device=device, dtype=dtype)
  return map_nested(data, op, sortkey=sortkey)

def unpack_structure(
  data: Any,
  sortkey: bool = False
) -> Tuple[Any, List[Any]]:
  """Extract structure and flattened data from a nested data
  For example:
    >>> data = {'a': 'abc', 'b': (2.0, [3, 4, 5])}
    >>> struct, flat_data = extract_struct(data)
    >>> flat_data
    ['abc', 2.0, [3, 4, 5]]
    >>> struct
    {'a': 0, 'b': (1, 2)}
  
  Args:
    data (Any): A nested data
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  _count_op = lambda v, c: next(c)
  counter = itertools.count(0)
  struct = map_nested(data, _count_op, counter, sortkey=sortkey)
  size = next(counter)
  flat_data = [None] * size
  def _flat_op(ind_and_data, flat_data):
    ind, data = ind_and_data
    flat_data[ind] = data
  map_nested_tuple((struct, data), _flat_op, flat_data, sortkey=sortkey)
  return struct, flat_data

def pack_sequence(
  struct: Any,
  flat_data: List[Any],
  sortkey: bool = False
) -> Any:
  """An inverse operation of `extract_structure`

  Args:
    struct (Any): A nested structure each data field contains
      an index of elements in `flat_data`
    flat_data (List[Any]): flattened data.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.
  """
  _struct_op = lambda ind, flat: flat[ind]
  data = map_nested(struct, _struct_op, flat_data, sortkey=sortkey)
  return data

def flatten_space(
  space: gym.spaces.Space,
  sortkey: bool = False
) -> Tuple[gym.spaces.Space]:
  """Flatten gym's nested space, like gym.Dict or gym.Tuple,
  to a tuple of spaces.

  Args:
    space (gym.spaces.Space): Nested or non-nested space.
    sortkey (bool): Whether to sort dict's key. Defaults
      to False.

  Returns:
    Tuple[gym.spaces.Space]: Flattened gym spaces.
  """
  return tuple(v for v in iter_nested_space(space, sortkey=sortkey))

def flatten(
  data: Any,
  sortkey: bool = False
) -> Tuple[Any]:
  """Flatten nested structure

  Args:
    data (Any): nested or non-nested data.
    sortkey (bool, optional): whether to sort dict's key.
      Defaults to False.
  
  Returns:
    Tuple[Any]: Flattened data.
  """
  return tuple(v for v in iter_nested(data, sortkey=sortkey))