# --- built in ---
from typing import (
  Any,
  DefaultDict,
  KeysView
)
from collections import defaultdict
from dataclasses import dataclass
# --- 3rd party ---
import omegaconf
# --- my module ---
from rlchemy.lib import utils as rl_utils

@dataclass
class LinkItem:
  link: str
  config: omegaconf.Container
  item: Any

class LinkStore(rl_utils.Singleton):
  store: DefaultDict[str, LinkItem] = defaultdict(dict)
  LinkSymbol: str = 'L'

  @classmethod
  def get_links(cls) -> KeysView:
    """Return all links"""
    return cls.store.keys()

  @classmethod
  def has_item(cls, link_name: str) -> bool:
    """Return true if the item is instantiated"""
    return link_name in cls.store.keys()
  
  @classmethod
  def store_item(
    cls,
    link_name: str,
    item: LinkItem,
    assert_unique: bool = True
  ):
    """Store an item into link store"""
    if assert_unique:
      assert not cls.has_item(link_name), \
        f"Item `{link_name}` is duplicated."
    cls.store[link_name] = item

  @classmethod
  def get_item(cls, link_name: str) -> LinkItem:
    assert cls.has_item(link_name), \
      f"Item `{link_name}` does not exist."
    return cls.store[link_name]