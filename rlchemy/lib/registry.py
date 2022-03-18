# --- bulit in ---
from collections import defaultdict
from typing import (
  Any,
  Dict,
  Optional,
  DefaultDict,
  List,
  Callable,
  Type
)
# --- 3rd party ---
# --- my module ---
from rlchemy.lib import utils as rl_utils


__all__ = [
  "register",
  "get"
]

class _RegistryEntry(dict):
  default: Type = None

class Registry(rl_utils.Singleton):
  entries: DefaultDict[str, Any] = defaultdict(_RegistryEntry)
  @classmethod
  def _register_impl(
    cls,
    _type: str,
    to_register: Optional[Any],
    alias: Optional[List[str]],
    default: Optional[bool] = False,
    overwrite: Optional[bool] = False,
    assert_type: Optional[Type] = None,
  ) -> Callable:
    def wrap(to_register):
      if assert_type is not None:
        # throw the exception if the object to register is not the
        # subclass of specified `assert_type`
        assert issubclass(to_register, assert_type), \
          f"{to_register} must be a subclass of {assert_type}"
      # formalize name list
      name_list = alias
      if not isinstance(name_list, (tuple, list)):
        name_list = [name_list]
      if hasattr(to_register, '__name__'):
        name = to_register.__name__
        if name not in name_list:
          name_list.append(name)
      for name in name_list:
        if not overwrite:
          # throw the exception for registry name conflicts
          reg = cls.entries[_type].get(name, None)
          assert name not in cls.entries[_type].keys(), \
            f"The name '{name}' is used by other registry {reg}" \
            "Set overwrite=True to ignore the conflicts"
        # ignore non-str names
        if isinstance(name, str):
          cls.entries[_type][name] = to_register
      # register as the default type
      if default:
        cls.entries[_type].default = to_register
        cls.entries[_type]['default'] = to_register
      return to_register
    
    if to_register is None:
      return wrap
    else:
      return wrap(to_register)

  @classmethod
  def register(
    cls,
    _type: str = '',
    alias: Optional[List[str]] = [],
    default: Optional[bool] = False,
    overwrite: Optional[bool] = False,
    assert_type: Optional[Type] = None
  ) -> Callable:
    """Register class

    For example:
    ```
    @Registry.register(
      'sche',
      alias = ['mysche'],
      default = True,
      assert_type = Scheduler
    )
    class MyScheduler(Scheduler):
      pass
    
    # Retrieve registered class
    assert (Registry.get('sche', 'MyScheduler')
              == Registry.get('sche', 'mysche'))

    # Retrieve default class
    assert (Registry.get('sche', 'MyScheduler')
              == Registry.get('sche'))
    ```

    If the class name or `alias` conflicts with the existing registry, it raises
    a key confliting exception. You can set `overwrite=True` to ignore the
    exception.

    Args:
      _type (str): type name of the class. Defaults to ''.
      alias (Optional[List[str]], optional): a list of other aliases.
        Defaults to [].
      default (Optional[bool], optional): whether to set as the default class of
        this `_type`. Defaults to False.
      overwrite (Optional[bool], optional): whether to overwrite the existing
        registries. Defaults to False.
      assert_type (Optional[Type], optional): raise an exception if the
        registering class is not a subclass of this type. Defaults to None.

    Returns:
      Callable: wrapping function.
    """
    return cls._register_impl(
      _type = _type,
      to_register = None,
      alias = alias,
      default = default,
      overwrite = overwrite,
      assert_type = assert_type
    )

  @classmethod
  def get(
    cls,
    _type: str = '',
    name: Optional[str] = None,
    default: Optional[Any] = None,
  ) -> Type:
    """Get registered class
    None is returned if the name is not in the registry.

    Args:
      _type (str): type name of the class. Defaults to ''.
      name (str): registered name to retrieval. None for retrieving
        default class.

    Returns:
      registered class, or None if the name is not in the registry.
    """
    if name is None:
      # return default registry
      return cls.entries[_type].default
    return cls.entries[_type].get(name, default)

  @classmethod
  def reset(cls):
    """Reset registries"""
    cls.entries = defaultdict(_RegistryEntry)

class Register(rl_utils.Singleton):
  """A helper class

  The followings are equivalent:
  ```
  @registry.registry.register('buffer', 'my_buffer')
  class MyBuffer():
    pass

  @registry.register('buffer', 'my_buffer')
  class MyBuffer():
    pass

  @registry.register.buffer('my_buffer')
  class MyBuffer():
    pass
  ```
  """
  def __getattr__(self, _type: str):
    def _wrap_register(*args, **kwargs):
      return Registry.register(_type, *args, **kwargs)
    return _wrap_register

  def __call__(self, *args, **kwargs):
    return Registry.register(*args, **kwargs)

class GetRegistry(rl_utils.Singleton):
  """A helper class

  The followings are equivalent:
  ```
  buffer_class = registry.registry.get('buffer', 'my_buffer')

  buffer_class = registry.get('buffer', 'my_buffer')

  buffer_class = registry.get.buffer('my_buffer')
  ```
  """
  def __getattr__(self, _type: str):
    def _wrap_get_registry(*args, **kwargs):
      return Registry.get(_type, *args, **kwargs)
    return _wrap_get_registry
  
  def __call__(self, *args, **kwargs):
    return Registry.get(*args, **kwargs)


registry = Registry.instance()
register = Register.instance()
get = GetRegistry.instance()