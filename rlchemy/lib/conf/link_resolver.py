# --- built in ---
import enum
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Union
)
from collections import defaultdict
from dataclasses import dataclass
# --- 3rd party ---
import omegaconf
from omegaconf import OmegaConf
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import registry as rl_registry
from rlchemy.lib.conf.link_store import LinkItem, LinkStore

__all__ = [
  'LinkResolver'
]

# === utils ===

TYPE_KEY: str = 'type'
TYPE_SPLIT: str = '/'

def default_instantiate_callback(pyconf: Any) -> Any:
  """The default callback to instantiate objects from registry
  You can customize instantiate callbacks by defining a method
  with the following annotaiton
  ```
  def callback(pyconf: Any) -> Tuple[Any, bool]
  ```
  The `bool` indicates whether the returned objects are valid.
  You can return `False`, then the next custom callback is called
  until to get `True`.
  By calling
  `LinkResolver.register_instantiate_callback('key', callback)`
  to register the callbacks. The 'key' can be any str used to
  indentify callbacks.

  Args:
    pyconf (Any): nested configurations in native python types.

  Returns:
    Any: created objects
  """
  if isinstance(pyconf, dict):
    item_type = pyconf.pop(TYPE_KEY, None)
    if item_type is None or not isinstance(item_type, str):
      return pyconf
    # the item type must be '{type}/{name}' or '{name}'
    *class_type, class_name = item_type.split(TYPE_SPLIT, 1)
    class_type = '' if not class_type else class_type[0]
    item_class = rl_registry.get(class_type, class_name)
    #TODO: use dataclasses to structured configurations
    return item_class(**pyconf)
  else:
    return pyconf

# === custom resolver ===

class ResolverState(enum.Enum):
  Instantiate = 1
  Retrieve = 2

class ResolverStateContext():
  def __init__(
    self,
    resolver: 'LinkResolver',
    state: ResolverState
  ):
    """A context object to temporarily set the LinkResolver's state
    when entering and set back to the original state on exit.

    Args:
        resolver (LinkResolver): link resolver instance.
        state (ResolverState): temporary state.
    """
    self.resolver = resolver
    self.orig_state = self.resolver.state
    self.resolver.state = state

  def __enter__(self):
    return self

  def __exit__(self, *args, **kwargs):
    self.resolver.state = self.orig_state


class LinkResolver(rl_utils.Singleton):
  """A custom omegaconf resolver to resolve interpolation tag
  `L` (by default). LinkResolver finds the omegaconf path to
  the specified item configuraitons and creates them (usually
  they are a user defined classes, nn.Module, ...etc)
  automatically. Note that this class should be used along with
  `rlchemy.lib.registry`. LinkResolver will retrieve the item's
  class and constructor from the registry.

  There are two modes (states) for LinkResolver
  * Instantiate: in this mode, LinkResolver only go through the
    whole omegaconf configurations and constructs (or we say
    instantiates) the items if the links to these items exist
    in the omegaconf. The items are created, but will not replace
    the original fields in omegaconf, since omegaconf only accepts
    primitive types, e.g. int, float, str, ...etc.
  * Retrieve: in contrast to Instantiate mode, LinkResolver will
    replace the fields with the created items in previous mode.
  
  Some example usages:
  1. Register classes, see `rlchemy.lib.registry`
    ```python
    @register('mlp')
    class MyMLP(nn.Module):
      def __init__(self, units: List[str]=[64, 64]):
        self.units = units
        ...

    @register.value('multihead')
    class MultiheadValueNets(nn.Module):
      def __init__(self, nets: List[nn.Module], n_heads: int=2):
        self.nets = nets
        self.n_heads = n_heads
        ...

    @register.policy('default')
    class PolicyNet(nn.Module):
      def __init__(self, net: nn.Module):
        self.net = net
        ...
    ```
  
  2. Create omegaconf from yaml string
    ```python
    conf = OmegaConf.create(
    '''
    agent:
      policy_net: ${L:model.policy_net}
      value_net: ${L:model.value_net}
    model:
      policy_net:
        type: "policy/default"
        net: ${L:model.policy_base}
      value_net:
        type: "value/multihead"
        net:
          - ${L:model.value_base}
          - ${L:model.value_base}
        n_heads: ${params.n_heads}
      policy_base:
        type: "mlp"
      value_base:
        type: "mlp"
    params:
      n_heads: 2
    '''
    )
    ```
    Note that ${L:model.policy_net} means we refer the instance of this
    item to the path conf.model.policy_net which is an item with type
    `PolicyNet` as we registered at step 1.
  3. Then the agent's `policy_net` and `value_net` are automatically
    created when we retrieve the values
    >>> print(conf.agent.policy_net)
    This outputs
    ```
    PolicyNet(
      (net): MyMLP()
    )
    ```
    
  4. Note that `conf.agent.value_net` is not created yet. If you want
    to create all the models at once, you can use `OmegaConf.resolve`
    ```python
    with LinkResolver.set_dont_resolve():
      OmegaConf.resolve(conf)
    ```
    the LinkResolver is set to `instantiate` mode under the
    `LinkResolver.set_dont_resolve` context. In this mode, the resolver
    creates all the models on calling `OmegaConf.resolve` methods and
    stores them in the LinkStore. It will not replace the original
    configurations with the created items as omegaconf throws errors if
    the elements in the configurations are not a primitive types supported
    by omegaconf, e.g. int, float, str, ...etc.
  
  5. If you want to get the complete configurations with created items and
    models, you can call by
    ```python
    with LinkResolver.set_resolve():
      py_conf = OmegaConf.to_container(conf, resolve=True)
    ```
    the LinkResolver is set to `retrieve` mode under the
    `LinkResolver.set_resolve` context. The returned object `py_conf`
    is a nested configuration in native python classes, e.g. dict, list and
    the created objects. In defaults, the LinkResolver is set to `retrieve`
    mode, so you can simply call by
    ```python
    py_conf = OmegaConf.to_container(conf, resolve=True)
    ```
  """
  state: ResolverState = ResolverState.Retrieve
  _default_instantiate_callback: Callable = default_instantiate_callback
  _custom_instantiate_callbacks: Dict[str, Callable] = dict()

  @classmethod
  def register_instantiate_callback(
    cls,
    key: str,
    callback: Callable,
    replace: bool = False
  ):
    if not replace:
      assert key not in cls._custom_instantiate_callbacks.keys()
    cls._custom_instantiate_callbacks[key] = callback

  @classmethod
  def register_custom_callback(cls, key: str, callable: Callable):
    cls._

  @classmethod
  def set_instantiate_callabck(cls, callable: Callable):
    cls._custom_instantiate_callback = callable

  @classmethod
  def is_state(cls, state: ResolverState) -> bool:
    """Check the resolver state is in `state`"""
    return cls.state == ResolverState(state)

  @classmethod
  def is_instantiate(cls) -> bool:
    return cls.is_state(ResolverState.Instantiate)

  @classmethod
  def is_retrieve(cls) -> bool:
    return cls.is_state(ResolverState.Retrieve)

  @classmethod
  def set_state(cls, state: ResolverState) -> ResolverStateContext:
    """Setting resolver's state"""
    return ResolverStateContext(cls, state)

  @classmethod
  def set_instantiate(cls) -> ResolverStateContext:
    return cls.set_state(ResolverState.Instantiate)

  @classmethod
  def set_retrieve(cls) -> ResolverStateContext:
    return cls.set_state(ResolverState.Retrieve)

  @classmethod
  def create_item(
    cls,
    pyconf: Any
  ):
    for callback in cls._custom_instantiate_callbacks.values():
      item, res = callback(pyconf)
      if res is True:
        return item
    return cls._default_instantiate_callback(pyconf)

  @classmethod
  def _instantiate_item_impl(
    cls,
    root_conf: omegaconf.Container,
    link_name: str,
    flags: List[str]
  ) -> LinkItem:
    # retrieve node configurations by link_name
    # Note that the link_name must be an absolute path from
    # root node. TODO: implement relative path.
    node_conf = OmegaConf.select(root_conf, link_name)
    # ensure that the nested nodes and values are all intepolated
    with cls.set_instantiate():
      OmegaConf.resolve(node_conf)
    # set resolver state to `Retrieve` to extract the nested
    # config structure into native python dict, list. And pass
    # it to instantiate items.
    with cls.set_retrieve():
      pyconf = OmegaConf.to_container(node_conf, resolve=True)
      # TODO: instantiate items
      # TODO: flag callbacks
      item = cls.create_item(pyconf)
      return LinkItem(link_name, node_conf, item)

  @classmethod
  def instantiate_item(
    cls,
    root_conf: omegaconf.Container,
    link_name: str,
    *flags: List[str]
  ) -> LinkItem:
    """Instantiate items by the given link and configurations

    Args:
        root_conf (omegaconf.Container): root configurations.
        link_name (str): absolute path to config node in dot-notation.

    Returns:
        LinkItem: an instantiated item.
    """
    return cls._instantiate_item_impl(root_conf, link_name, flags)

  @classmethod
  def get_or_instantiate_item(
    cls,
    root_conf: omegaconf.Container,
    link_name: str,
    *flags: List[str]
  ) -> LinkItem:
    """Get instantiated item or create a new one.

    Args:
        root_conf (omegaconf.Container): root configurations.
        link_name (str): absolute path to config node in dot-notation.

    Returns:
        LinkItem: an instantiated item.
    """
    raise NotImplementedError

  def __call__(
    self,
    link_name: str,
    *flags: List[str],
    _parent_: omegaconf.Container,
    _root_: omegaconf.Container
  ) -> Union[str, Any]:
    """Main resolver call, called by OmegaConf.resolve

    Args:
        link_name (str): absolute path to config node in dot-notation.
        _parent_ (omegaconf.Container): parent node.
        _root_ (omegaconf.Container): root node.

    Returns:
        Union[str, Any]: return the original interpolation tags (str)
          in instantiate mode. Otherwise, return the instantiated item.
    """
    link_store = LinkStore.instance()
    if not link_store.has_item(link_name):
      # Create a new item if it is not instantiated yet
      item = self.instantiate_item(
        root_conf = _root_,
        link_name = link_name,
        *flags
      )
      # throw assertion error if `link_name` already exists in the
      # link store
      link_store.store_item(link_name, item, assert_unique=True)
    if self.is_retrieve():
      # return the instantiated item
      item = link_store.get_item(link_name)
      # TODO handle flags, e.g. deepcopy
      return item.item
    else:
      # return the original interpolation tags ${L:{full_link}}
      # e.g. ${L:object,deepcopy}
      full_link = ','.join([link_name] + list(flags))
      return (
          '${'
        + link_store.LinkSymbol
        + f':{full_link}'
        + '}'
      )

  @classmethod
  def register_resolver(
    cls,
    replace: bool = True,
    **kwargs
  ):
    OmegaConf.register_new_resolver(
      LinkStore.LinkSymbol,
      cls.instance(),
      replace = replace,
      **kwargs
    )

  @classmethod
  def set_resolve(cls):
    return cls.set_retrieve()

  @classmethod
  def set_dont_resolve(cls):
    return cls.set_instantiate()

link_store = LinkStore()
link_resolver = LinkResolver()

LinkResolver.register_resolver()
