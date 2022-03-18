# --- built in ---
import os
import json
import base64
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Optional,
)
# --- 3rd party ---
import gym
import cloudpickle
import numpy as np
from atomicwrites import atomic_write
# --- my module ---
from rlchemy.lib.utils import nest as rl_nest

__all__ = [
  'spaces',
  'StateObject',
  'is_image_space',
  'flatten_dicts',
  'is_bounded',
  'safe_makedirs',
  'atomic_write_file',
  'read_from_file',
  'encode_base64',
  'decode_base64',
  'is_json_serializable',
  'to_json_serializable',
  'from_json_serializable',
  'safe_json_dumps',
  'safe_json_loads',
  'safe_json_dump',
  'safe_json_load',
]

# === Const params ===
SERIALIZATION_KEY='#RLCHENY_SERIALIZED'

# Extra space profiles
class spaces:
  All = [gym.spaces.Box, gym.spaces.Discrete,
       gym.spaces.MultiBinary, gym.spaces.MultiDiscrete,
       gym.spaces.Tuple, gym.spaces.Dict]
  NonNested = [gym.spaces.Box, gym.spaces.Discrete,
         gym.spaces.MultiBinary, gym.spaces.MultiDiscrete]
  Nested = [gym.spaces.Tuple, gym.spaces.Dict]

class StateObject(dict):
  """An object-like dictionary that you can get/set
  items/attributes by either __getitem__/__setitem__
  or __getattr__/__setattr__.
  For example:
  >>> d = StateObject()
  >>> d['abc'] = 10
  >>> print(d['abc'])
  This is equivalent to
  >>> d.abc = 10
  >>> print(d.abc)
  Also, StateObject can be serialized by JSON by calling
  `tostring` and `fromstring`.
  """
  def __new__(cls, *args, **kwargs):
    self = super().__new__(cls, *args, **kwargs)
    self.__dict__ = self
    return self
  
  def to_json(self, indent: Optional[int]=None) -> str:
    return safe_json_dumps(self, indent=indent)

  @classmethod
  def from_json(cls, string: str) -> 'StateObject':
    self = StateObject()
    self.update(safe_json_loads(string))
    return self

  def tostring(self, indent: Optional[int]=None) -> str:
    #TODO: deprecated
    return self.to_json(indent=indent)

  @classmethod
  def fromstring(cls, string: str) -> 'StateObject':
    #TODO: deprecated
    return cls.from_json(string=string)

# === utils ===

def flatten_dicts(dicts: List[Dict]) -> Dict:
  """Flatten a list of dicts

  Args:
    dicts (list): list of dicts

  Returns:
    dict: flattened dict
  """
  agg_dict = {}
  for d in dicts:
    for k, v in d.items():
      agg_dict.setdefault(k, []).append(v)
  return agg_dict

def is_image_space(space: gym.spaces.Space) -> bool:
  return (isinstance(space, gym.spaces.Box)
    and np.dtype(space.dtype) == np.uint8
    and len(space.shape) == 3)

def is_bounded(space: gym.spaces.Space) -> bool:
  if isinstance(space, gym.spaces.Box):
    return (not np.any(np.isinf(space.low))
      and not np.any(np.isinf(space.high))
      and np.all(space.high-space.low > 0.))
  return True

def is_batch_sample(
  inputs: np.ndarray,
  space: Optional[gym.spaces.Space] = None,
  rank: int = None,
  validate: bool = False
):
  """Check if it's a batch of samples or a single sample of the space

  Args:
    inputs (np.ndarray): input samples, can be a nested type.
    space (gym.Space, optional): gym space, can be a nested space.
    rank (int): expected sample rank.
    validate (bool): validate `inputs`.

  Returns:
    bool: True if it's a batch. False, otherwise.
  """
  if validate:
    raise NotImplementedError
  first_input = next(iter(rl_nest.iter_nested(inputs)))
  if space is not None:
    first_space = next(iter(rl_nest.iter_nested_space(space)))
    return len(first_input.shape) > len(first_space.shape)
  elif rank is not None:
    return len(first_input.shape) > rank
  else:
    raise ValueError("Either `space` or `rank` must be specified.")

# === file utils ===

def safe_makedirs(dirpath: Optional[str]=None, filepath: Optional[str]=None):
  if dirpath:
    os.makedirs(dirpath, exist_ok=True)
  if filepath:
    safe_makedirs(dirpath=os.path.dirname(filepath))

def atomic_write_file(filepath: str, string: str, mkdir: bool=True):
  if mkdir:
    safe_makedirs(filepath=filepath)
  with atomic_write(filepath, overwrite=True) as f:
    f.write(string)

def read_from_file(filepath: str) -> str:
  with open(filepath, 'r') as f:
    return f.read()

def read_file_to_string(filepath: str) -> str:
  return read_from_file(filepath)

# === json utils ===

def is_json_serializable(obj: Any) -> bool:
  """Check if the object is json serializable

  Args:
    obj (Any): object

  Returns:
    bool: True for a json serializable object
  """
  try:
    json.dumps(obj, ensure_ascii=False)
    return True
  except:
    return False

def encode_base64(obj: Any) -> str:
  """Encode object to base64"""
  encoded_obj = base64.b64encode(
        cloudpickle.dumps(obj)).decode()
  return encoded_obj

def decode_base64(encoded_obj: str) -> Any:
  """Decode encoded object from base64"""
  obj = cloudpickle.loads(
        base64.b64decode(encoded_obj.encode()))
  return obj

def to_json_serializable(obj: Any) -> Any:
  """Encode any object to a json serializable object

  Args:
    obj (Any): object

  Returns:
    Any: json serializable object
  """
  if not is_json_serializable(obj):
    encoded_obj = encode_base64(obj)
    encoded_obj = {SERIALIZATION_KEY: encoded_obj}
  else:
    encoded_obj = obj

  return encoded_obj

def from_json_serializable(encoded_obj: Any) -> Any:
  """Decode any encoded json serializable object
  if the object is not encoded, then do nothing.

  Args:
    encoded_obj (Any): encoded json serializable object

  Returns:
    Any: decoded object
  """
  if (isinstance(encoded_obj, dict)
      and SERIALIZATION_KEY in encoded_obj.keys()):
    obj = encoded_obj[SERIALIZATION_KEY]
    obj = decode_base64(obj)
  else:
    obj = encoded_obj
  return obj

def safe_json_dumps(
  obj: Any,
  indent: int = 2,
  ensure_ascii: bool = False,
  default: Callable = to_json_serializable,
  **kwargs
) -> str:
  string = json.dumps(
    obj,
    indent = indent,
    ensure_ascii = ensure_ascii,
    default = default,
    **kwargs
  )
  return string

def safe_json_loads(
  string: str,
  object_hook: Callable = from_json_serializable,
  **kwargs
) -> Any:
  obj = json.loads(
    string,
    object_hook = object_hook,
    **kwargs
  )
  return obj

def safe_json_dump(
  filepath: str,
  obj: Any,
  **kwargs
):
  string = safe_json_dumps(obj, **kwargs)
  atomic_write_file(filepath, string)

def safe_json_load(
  filepath: str,
  **kwargs
) -> Any:
  obj = None
  if os.path.isfile(filepath):
    string = read_from_file(filepath)
    obj = safe_json_loads(string, **kwargs)
  return obj
