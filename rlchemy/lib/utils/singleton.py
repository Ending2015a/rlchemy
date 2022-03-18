# --- built in ---
from typing import (
  Any,
  Dict,
)
# --- 3rd party ---
# --- my module ---

__all__ = [
  'Singleton'
]

class SingletonMeta(type):
  _instances: Dict["SingletonMeta", "SingletonMeta"] = {}
  def __call__(cls, *args: Any, **kwargs: Any) -> Any:
    if cls not in cls._instances:
      cls._instances[cls] = super(SingletonMeta, cls).__call__(
        *args, **kwargs
      )
    return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
  @classmethod
  def instance(cls, *args: Any, **kwargs: Any) -> Any:
    return cls(*args, **kwargs)