# --- built in ---
from typing import List, Callable
# --- 3rd party ---
import omegaconf
from omegaconf import OmegaConf
# --- my module ---
from rlchemy.lib import utils as rl_utils


def register_resolver(name: str, **kwargs):
  """A helper function for registering resolvers"""
  def _wrap(resolver: Callable):
    OmegaConf.register_new_resolver(
      name, resolver, **kwargs
    )
  return _wrap


# === resolvers ===

@register_resolver('range', replace=True)
def resolver_range(
  *args: List[int],
  _parent_: omegaconf.Container
) -> omegaconf.Container:
  """Range resolver, create a list config from python range

  Example:
  >>> conf = OmegaConf.create('a: ${range:1, 3}')
  >>> conf.a
  [1, 2]

  Returns:
    omegaconf.Container: ListConfig
  """
  return OmegaConf.create(list(range(*args)), parent=_parent_)