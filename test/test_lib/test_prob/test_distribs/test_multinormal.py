# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
from scipy import stats as sp_stats
from parameterized import parameterized

# --- my module ---
from rlchemy.lib.prob import distribs as rl_distribs
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

class TestDistribsModuleMultiNormal(TestCase):
  def test_shape_exception(self):
    mu = 1.
    sigma = -5.
    with self.assertRaises(RuntimeError):
      rl_distribs.MultiNormal(mean=mu, scale=sigma)

  def test_shape_no_exception(self):
    mu = [1.]
    sigma = [-5.]
    rl_distribs.MultiNormal(mean=mu, scale=sigma)

  @parameterized.expand([
    ([1], [1]), 
    ([2, 3, 4], [1, 1, 4]),
    ([2, 3, 4], [1]),
    ([2, 3, 4], []),
    ([1, 1, 4], [2, 3, 4]),
    ([1], [2, 3, 4]),
    ([], [2, 3, 4])
  ])
  def test_shapes(self, mean_shape, scale_shape):
    full_shape = (np.ones(mean_shape) * np.ones(scale_shape)).shape
    batch_shape = full_shape[:-1]
    event_shape = full_shape[-1:]
    dist = make_multinormal(mean_shape, scale_shape, dtype=torch.float32)
    self.assertEqual(1, dist.event_ndims)
    self.assertArrayEqual(full_shape, dist.shape)
    self.assertArrayEqual(batch_shape, dist.batch_shape)
    self.assertArrayEqual(event_shape, dist.event_shape)
    self.assertArrayEqual(mean_shape, dist.mean.shape)
    self.assertArrayEqual(scale_shape, dist.scale.shape)
    self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(full_shape)).shape)
    self.assertArrayEqual(full_shape, dist.mode().shape)
    self.assertArrayEqual(full_shape, dist.sample().shape)
    self.assertArrayEqual(batch_shape, dist.entropy().shape)
    dist2 = make_multinormal(mean_shape, scale_shape, dtype=torch.float32)
    self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)

  @parameterized.expand([
    (torch.float32,),
    (torch.float64,),
  ])
  def test_dtypes(self, dtype):
    dist = make_multinormal([1], [1], dtype=dtype)
    self.assertEqual(dtype, dist.dtype)
    self.assertEqual(dtype, dist.mean.dtype)
    self.assertEqual(dtype, dist.scale.dtype)
    self.assertEqual(dtype, dist.log_prob(0).dtype)
    self.assertEqual(dtype, dist.mode().dtype)
    self.assertEqual(dtype, dist.sample().dtype)
    self.assertEqual(dtype, dist.entropy().dtype)
    dist2 = make_multinormal([1], [1], dtype=dtype)
    self.assertEqual(dtype, dist.kl(dist2).dtype)

  def test_prob(self):
    mu = np.asarray([1.0, -1.0], dtype=np.float32)
    sigma = np.asarray([3.0, 2.0], dtype=np.float32)
    x = np.array([2.5, 0.5], dtype=np.float32)
    dist = rl_distribs.MultiNormal(mean=mu, scale=sigma)
    # test mean scale
    self.assertArrayEqual(mu, dist.mean)
    self.assertArrayEqual(sigma, dist.scale)
    # test prob, log_prob
    exp_mvn = sp_stats.multivariate_normal(mu, np.diag(sigma)**2)
    self.assertArrayClose(exp_mvn.logpdf(x), dist.log_prob(x))
    self.assertArrayClose(np.exp(exp_mvn.logpdf(x)), dist.prob(x))

  def test_sample(self):
    mu = np.asarray([1.0, -1.0])
    sigma = np.asarray([1.0, 5.0])
    dist = rl_distribs.MultiNormal(mean=mu, scale=sigma)
    set_test_seed()
    draws = np.asarray(dist.sample(500000))
    self.assertArrayEqual(draws.shape, (500000,2))
    self.assertAllClose(draws.mean(axis=0), mu, atol=1e-1)
    self.assertAllClose(draws.var(axis=0), sigma**2, atol=1e-1)

  def test_entropy(self):
    mu = np.asarray([1.0, 0.0, -1.0])
    sigma = np.asarray([1.0, 2.0, 3.0])
    dist = rl_distribs.MultiNormal(mean=mu, scale=sigma)
    exp_mn = sp_stats.multivariate_normal(mean=mu, cov=np.diag(sigma)**2)
    self.assertArrayEqual(exp_mn.entropy().shape, dist.entropy().shape)
    self.assertAllClose(exp_mn.entropy(), dist.entropy())

  def test_kl(self):
    mu_a = np.array([3.0, -1.0])
    sigma_a = np.array([1.0, 2.5])
    mu_b = np.array([-3.0, 1.5])
    sigma_b = np.array([0.5, 1.0])
    dist_a = rl_distribs.MultiNormal(mean=mu_a, scale=sigma_a)
    dist_b = rl_distribs.MultiNormal(mean=mu_b, scale=sigma_b)
    kl = dist_a.kl(dist_b)
    expected_kl = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
      (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b))).sum()
    self.assertArrayEqual(kl.shape, [])
    self.assertAllClose(expected_kl, kl)
    # test estimate kl
    set_test_seed()
    draws = dist_a.sample(100000)
    sample_kl = dist_a.log_prob(draws) - dist_b.log_prob(draws)
    sample_kl = torch.mean(sample_kl, dim=0)
    self.assertAllClose(expected_kl, sample_kl, atol=0.0, rtol=1e-2)