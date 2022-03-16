# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
# --- my module ---
from rlchemy.lib.utils import th_common as rl_th_common
from test.utils import TestCase

class TestUtilsThCommonModule(TestCase):
  """Test rlchemy.lib.utils.th_common module"""

  def test_set_seed(self):
    rl_th_common.set_seed(1)
    a = np.random.normal(size=(3,))
    rl_th_common.set_seed(1)
    b = np.random.normal(size=(3,))
    self.assertArrayEqual(a, b)
  
  def test_to_tensor(self):
    a = np.array([1,2,3])
    a_th = rl_th_common.to_tensor(a)
    self.assertTrue(torch.is_tensor(a_th))
    self.assertTrue(rl_th_common.to_tensor(a_th) is a_th)
    a = [1,2,3]
    a_th = rl_th_common.to_tensor(a)
    self.assertTrue(torch.is_tensor(a_th))
  
  def test_to_numpy(self):
    a_th = torch.tensor([1,2,3])
    a = rl_th_common.to_numpy(a_th)
    self.assertTrue(isinstance(a, np.ndarray))
    a = rl_th_common.to_numpy([1, 2, 3])
    self.assertTrue(isinstance(a, np.ndarray))

  def test_to_tensor_like(self):
    a_th = torch.tensor([1,2,3], dtype=torch.float32)
    b_th = torch.tensor([1,2,3], dtype=torch.int64)
    b_th = rl_th_common.to_tensor_like(b_th, a_th)
    self.assertTrue(b_th.dtype == a_th.dtype)

  def test_normalize_denormalize(self):
    x = np.arange(21, dtype=np.float32)*0.1
    y = np.arange(21, dtype=np.float32)*0.05
    x_ = rl_th_common.normalize(x, x.min(), x.max())
    self.assertArrayClose(x_, y, decimal=6)
    y_ = rl_th_common.denormalize(y, x.min(), x.max())
    self.assertArrayClose(x, y_, decimal=6)

  def test_normalize_denormalize_tensor(self):
    x = torch.arange(21, dtype=torch.float32)*0.1
    y = torch.arange(21, dtype=torch.float32)*0.05
    x_ = rl_th_common.normalize(x, x.min(), x.max())
    self.assertArrayClose(x_, y, decimal=6)
    y_ = rl_th_common.denormalize(y, x.min(), x.max())
    self.assertArrayClose(x, y_, decimal=6)
