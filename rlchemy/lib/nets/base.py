# --- built in ---
from typing import (
  Any,
  Tuple,
  Iterable,
  Union,
  Optional
)
# --- 3rd party ---
import torch
from torch import nn
# --- my module ---

__all__ = [
  'DelayedModule',
  'Parallel'
]

class DelayedModule(nn.Module):
  def __init__(self):
    """DelayedModule allows delayed module creation
    The function `build()` is called at the first time forwarding
    this module if the module is not built. Users can implement
    the module building procedure in `build()`.

    For example, the following implements a delayed module which
    only allows the delayed module creation:
    ```
    class MyDelayedModule(DelayedModule):
      def __init__(self, unit=64):
        super().__init__()
        self.unit = unit
      
      def build(self, input_shape: torch.Size):
        in_dim = input_shape[-1]
        out_dim = self.unit
        self.model = nn.Linear(in_dim, out_dim)

      def forward(self, x):
        return self.model(x)
    ```

    Another example implements both delayed and non-delayed
    module creation, which can be compatible with pytorch-style
    module:
    ```
    class MyDelayedModule(DelayedModule):
      def __init__(self, dim=None, unit=64):
        super().__init__()
        self.unit = unit
        if dim is not None:
          self.build(torch.Size([dim]))
          self.mark_as_built()
      
      def build(self, input_shape: torch.Size):
        in_dim = input_shape[-1]
        out_dim = self.unit
        self.model = nn.Linear(in_dim, out_dim)

      def forward(self, x):
        return self.model(x)
    ```
    Note that you need to call `mark_as_built()` after building
    the module. This avoids repeated creation of the module.

    A more complicated example is your module has a complex input
    tensors, e.g. nested tensors, multiple input tensors. You can
    customize `get_input_shapes()` method, which returns the tensor
    shapes and passed to the `build()` for module construction.
    ```
    class MyComplexDelayedModule(DelayedModule):
      def __init__(self, unit=64):
        super().__init__()
        self.unit = unit

      def get_input_shapes(self, x, y):
        return (x.shape, y.shape)

      def build(self, input_shapes: Tuple[torch.Size]):
        x_shape, y_shape = input_shapes
        self.model_x = nn.Linear(x_shape[-1], self.unit)
        self.model_y = nn.Linear(y_shape[-1], self.unit)

      def forward(self, x, y):
        out_x = self.model_x(x)
        out_y = self.model_y(y)
        return torch.cat([out_x, out_y], dim=-1)
    ```
    """
    super().__init__()
    self._built: bool = False

  @property
  def built(self) -> bool:
    """Return True if the module is already built"""
    return self._built

  def build(self, input_shape: torch.Size):
    """Customize your module building here"""
    pass

  def mark_as_built(self):
    """Mark the module as already built
    This must be called after the module is built to
    prevent from repeated creation of the module.
    """
    self._built = True

  def get_input_shapes(
    self,
    x: torch.Tensor,
    *args,
    **kwargs
  ) -> Union[torch.Size, Any]:
    """Get the input tensor shape
    You can customize this function if you have a complicated
    tensor shape, e.g. nested inputs.
    
    Args:
      x (torch.Tensor): input tensors.

    Returns:
      Union[torch.Size, Any]: usually this function returns a
        torch.Size, which is then passed into `build()` function.
        However, if you have a complicated tensor shape, e.g.
        nested inputs or multiple input tensors, you can return
        anything you want.
    """
    return x.shape

  def _maybe_build(self, x: torch.Tensor, *args, **kwargs):
    """Build the module if the module is not built"""
    if not self.built:
      shapes = self.get_input_shapes(x, *args, **kwargs)
      self.build(shapes)
      self.mark_as_built()
  
  def __call__(self, x: torch.Tensor, *args, **kwargs) -> Any:
    """Create module before forwarding"""
    self._maybe_build(x, *args, **kwargs)
    return super().__call__(x, *args, **kwargs)


class Parallel(nn.ModuleList):
  def __init__(
    self,
    modules: Optional[Iterable[nn.Module]] = None
  ):
    """A module container which forward the input tensors into
    all of the submodules.

    Args:
      modules (Iterable[nn.Module], optional): submodules. Defaults
        to None.
    """
    super().__init__(modules)

  def forward(
    self,
    x: torch.Tensor,
    *args,
    **kwargs
  ) -> Tuple[torch.Tensor, ...]:
    """Forward each submodules and return the results tensors
    as a tuple.

    Args:
      x (torch.Tensor): input tensors.

    Returns:
      Tuple[torch.Tensor, ...]: a tuple of output tensors from
        each submodule.
    """
    return tuple(
      module(x, *args, **kwargs) for module in self
    )
