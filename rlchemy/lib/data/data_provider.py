# --- built in ---
from typing import Callable, Iterator
# --- 3rd party ---
import torch
from torch.utils.data import IterableDataset
# --- my module ---

__all__ = [
  'BatchProvider'
]

class BatchProvider(IterableDataset):
  def __init__(
    self,
    generate_batch_fn: Callable
  ):
    self.generate_batch_fn = generate_batch_fn

  def __iter__(self) -> Iterator:
    return self.generate_batch_fn