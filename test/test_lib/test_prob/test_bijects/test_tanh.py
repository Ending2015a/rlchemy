# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
from parameterized import parameterized
# --- my module ---
from rlchemy.lib.prob import distribs as rl_distribs
from rlchemy.lib.prob import bijects as rl_bijects
from rlchemy.lib import utils as rl_utils
from test.utils import TestCase

TEST_SEED = 1

def set_test_seed():
  rl_utils.set_seed(TEST_SEED)

def get_test_seed():
  return TEST_SEED

def make_normal(mean_shape, scale_shape, dtype=torch.float32, seed=get_test_seed()):
  rl_utils.set_seed(seed)
  mean = torch.zeros(mean_shape).uniform_(-10, 10)
  scale = torch.zeros(scale_shape).uniform_(-10, 10)
  return rl_distribs.Normal(mean.float(), scale.float(), dtype=dtype)

def make_multinormal(mean_shape, scale_shape, dtype=torch.float32, seed=get_test_seed()):
  rl_utils.set_seed(seed)
  mean = torch.zeros(mean_shape).uniform_(-10, 10)
  scale = torch.zeros(scale_shape).uniform_(-10, 10)
  return rl_distribs.MultiNormal(mean.float(), scale.float(), dtype=dtype)


class TestBijectsModuleTanh(TestCase):
  def test_bijector_init_exception(self):
    with self.assertRaises(ValueError):
      rl_bijects.Tanh(object())

  def test_bijector_init_no_exception(self):
    dist = rl_distribs.Normal(mean=1.0, scale=2.0)
    dist = rl_bijects.Tanh(dist)
  
  @parameterized.expand([
    ([], []),
    ([1], [1]),
    ([2, 3, 4], [1, 1, 4]),
    ([2, 3, 4], [1]),
    ([2, 3, 4], []),
    ([1, 1, 4], [2, 3, 4]),
    ([1], [2, 3, 4]),
    ([], [2, 3, 4])
  ])
  def test_shapes_normal(self, mean_shape, scale_shape):
    batch_shape = torch.broadcast_shapes(torch.Size(mean_shape),
                       torch.Size(scale_shape))
    dist = make_normal(mean_shape, scale_shape, dtype=torch.float32)
    dist = rl_bijects.Tanh(dist)
    self.assertEqual(0, dist.event_ndims)
    self.assertArrayEqual(batch_shape, dist.shape)
    self.assertArrayEqual(batch_shape, dist.batch_shape)
    self.assertArrayEqual([], dist.event_shape)
    self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(batch_shape)).shape)
    self.assertArrayEqual(batch_shape, dist.mode().shape)
    self.assertArrayEqual(batch_shape, dist.sample().shape)

  @parameterized.expand([
    (torch.float32,),
    (torch.float64,),
  ])
  def test_dtypes_normal(self, dtype):
    dist = make_normal([], [], dtype=dtype)
    self.assertEqual(dtype, dist.dtype)
    self.assertEqual(dtype, dist.log_prob(0).dtype)
    self.assertEqual(dtype, dist.mode().dtype)
    self.assertEqual(dtype, dist.sample().dtype)

  @parameterized.expand([
    ([1], [1]),
    ([2, 3, 4], [1, 1, 4]),
    ([2, 3, 4], [1]),
    ([2, 3, 4], []),
    ([1, 1, 4], [2, 3, 4]),
    ([1], [2, 3, 4]),
    ([], [2, 3, 4])
  ])
  def test_shapes_multinormal(self, mean_shape, scale_shape):
    full_shape = (np.ones(mean_shape) * np.ones(scale_shape)).shape
    batch_shape = full_shape[:-1]
    event_shape = full_shape[-1:]
    dist = make_multinormal(mean_shape, scale_shape, dtype=torch.float32)
    dist = rl_bijects.Tanh(dist)
    self.assertEqual(1, dist.event_ndims)
    self.assertArrayEqual(full_shape, dist.shape)
    self.assertArrayEqual(batch_shape, dist.batch_shape)
    self.assertArrayEqual(event_shape, dist.event_shape)
    self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(full_shape)).shape)
    self.assertArrayEqual(full_shape, dist.mode().shape)
    self.assertArrayEqual(full_shape, dist.sample().shape)

  @parameterized.expand([
    (torch.float32,),
    (torch.float64,),
  ])
  def test_dtypes_multinormal(self, dtype):
    dist = make_multinormal([1], [1], dtype=dtype)
    dist = rl_bijects.Tanh(dist)
    self.assertEqual(dtype, dist.dtype)
    self.assertEqual(dtype, dist.log_prob(0).dtype)
    self.assertEqual(dtype, dist.mode().dtype)
    self.assertEqual(dtype, dist.sample().dtype)

  def test_tanh(self):
    dist = make_normal([], [], dtype=torch.float64)
    dist_tanh = rl_bijects.Tanh(dist)
    x = np.linspace(-3., 3., 100).reshape([2, 5, 10]).astype(np.float64)
    y = np.tanh(x)
    ildj = -np.log1p(-np.square(np.tanh(x)))
    self.assertAllClose(y, dist_tanh.forward(x), atol=0., rtol=1e-6)
    self.assertAllClose(x, dist_tanh.inverse(y), atol=0., rtol=1e-4)
    self.assertAllClose(-ildj, dist_tanh.log_det_jacob(x), atol=0., rtol=1e-4)