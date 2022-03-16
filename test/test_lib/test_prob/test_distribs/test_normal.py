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

class TestDistribsModuleNormal(TestCase):
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
  def test_shapes(self, mean_shape, scale_shape):
    batch_shape = torch.broadcast_shapes(torch.Size(mean_shape),
                      torch.Size(scale_shape))
    dist = make_normal(mean_shape, scale_shape, dtype=torch.float32)
    self.assertEqual(0, dist.event_ndims)
    self.assertArrayEqual(batch_shape, dist.shape)
    self.assertArrayEqual(batch_shape, dist.batch_shape)
    self.assertArrayEqual([], dist.event_shape)
    self.assertArrayEqual(mean_shape, dist.mean.shape)
    self.assertArrayEqual(scale_shape, dist.scale.shape)
    self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(batch_shape)).shape)
    self.assertArrayEqual(batch_shape, dist.mode().shape)
    self.assertArrayEqual(batch_shape, dist.sample().shape)
    self.assertArrayEqual(batch_shape, dist.entropy().shape)
    dist2 = make_normal(mean_shape, scale_shape, dtype=torch.float32)
    self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)

  @parameterized.expand([
    (torch.float32,),
    (torch.float64,),
  ])
  def test_dtypes(self, dtype):
    dist = make_normal([], [], dtype=dtype)
    self.assertEqual(dtype, dist.dtype)
    self.assertEqual(dtype, dist.mean.dtype)
    self.assertEqual(dtype, dist.scale.dtype)
    self.assertEqual(dtype, dist.log_prob(0).dtype)
    self.assertEqual(dtype, dist.mode().dtype)
    self.assertEqual(dtype, dist.sample().dtype)
    self.assertEqual(dtype, dist.entropy().dtype)
    dist2 = make_normal([], [], dtype=dtype)
    self.assertEqual(dtype, dist.kl(dist2).dtype)

  def test_prob(self):
    batch_size = 6
    mu = np.asarray([3.0] * batch_size, dtype=np.float32)
    sigma = np.asarray([np.sqrt(10.0)] * batch_size, dtype=np.float32)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    # test mean scale
    self.assertArrayEqual(mu, dist.mean)
    self.assertArrayEqual(sigma, dist.scale)
    # test prob, log_prob
    expected_log_prob = sp_stats.norm(mu, sigma).logpdf(x)
    self.assertArrayClose(expected_log_prob, dist.log_prob(x))
    self.assertArrayClose(np.exp(expected_log_prob), dist.prob(x))

  def test_prob_multidims(self):
    batch_size = 6
    mu = np.asarray([[3.0, -3.0]] * batch_size, dtype=np.float32)
    sigma = np.asarray(
      [[np.sqrt(10.0), np.sqrt(15.0)]] * batch_size, dtype=np.float32)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    # test mean scale
    self.assertArrayEqual(mu, dist.mean)
    self.assertArrayEqual(sigma, dist.scale)
    # test prob, log_prob
    expected_log_prob = sp_stats.norm(mu, sigma).logpdf(x)
    self.assertArrayClose(expected_log_prob, dist.log_prob(x))
    self.assertArrayClose(np.exp(expected_log_prob), dist.prob(x))

  def test_mode(self):
    batch_size = 6
    mu = np.asarray([3.0] * batch_size, dtype=np.float32)
    sigma = np.asarray([np.sqrt(10.0)] * batch_size, dtype=np.float32)
    x = np.array([-2.5, 2.5, 4.0, 0.0, -1.0, 2.0], dtype=np.float32)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    # test prob, log_prob
    self.assertArrayEqual(mu.shape, dist.mode().shape)
    self.assertArrayClose(mu, dist.mode())

  def test_mode_multidims(self):
    batch_size = 6
    mu = np.asarray([[3.0, -3.0]] * batch_size, dtype=np.float32)
    sigma = np.asarray(
      [[np.sqrt(10.0), np.sqrt(15.0)]] * batch_size, dtype=np.float32)
    x = np.array([[-2.5, 2.5, 4.0, 0.0, -1.0, 2.0]], dtype=np.float32).T
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    # test prob, log_prob
    self.assertArrayEqual(mu.shape, dist.mode().shape)
    self.assertArrayClose(mu, dist.mode())

  def test_sample(self):
    mu = np.asarray(3.0)
    sigma = np.sqrt(3.0)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    set_test_seed()
    draws = np.asarray(dist.sample(100000))
    self.assertArrayEqual(draws.shape, (100000,))
    self.assertAllClose(draws.mean(), mu, atol=1e-1)
    self.assertAllClose(draws.std(), sigma, atol=1e-1)

  def test_sample_with_batch(self):
    batch_size = 2
    mu = np.asarray([[3.0, -3.0]] * batch_size)
    sigma = np.asarray([[np.sqrt(2.0), np.sqrt(3.0)]] * batch_size)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    set_test_seed()
    draws = np.asarray(dist.sample(100000))
    self.assertArrayEqual(draws.shape, (100000, batch_size, 2))
    self.assertAllClose(draws[:, 0, 0].mean(), mu[0, 0], atol=1e-1)
    self.assertAllClose(draws[:, 0, 0].std(), sigma[0, 0], atol=1e-1)
    self.assertAllClose(draws[:, 0, 1].mean(), mu[0, 1], atol=1e-1)
    self.assertAllClose(draws[:, 0, 1].std(), sigma[0, 1], atol=1e-1)
  
  def test_sample_multidims(self):
    mu = np.asarray(3.0)
    sigma = np.sqrt(3.0)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    set_test_seed()
    draws = np.asarray(dist.sample([100, 1000]))
    self.assertArrayEqual(draws.shape, (100, 1000))
    self.assertAllClose(draws.mean(), mu, atol=1e-1)
    self.assertAllClose(draws.std(), sigma, atol=1e-1)

  def test_entropy(self):
    mu = np.asarray(2.34)
    sigma = np.asarray(4.56)
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    self.assertArrayEqual((), dist.entropy().shape)
    self.assertAllClose(sp_stats.norm(mu, sigma).entropy(), dist.entropy())

  def test_entropy_multidims(self):
    mu = np.asarray([1.0, 1.0, 1.0])
    sigma = np.asarray([[1.0, 2.0, 3.0]]).T
    dist = rl_distribs.Normal(mean=mu, scale=sigma)
    expected_ent = 0.5 * np.log(2 * np.pi * np.exp(1) * (mu*sigma)**2)
    self.assertArrayEqual(expected_ent.shape, dist.entropy().shape)
    self.assertAllClose(expected_ent, dist.entropy())

  def test_kl(self):
    batch_size = 6
    mu_a = np.array([3.0] * batch_size)
    sigma_a = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    mu_b = np.array([-3.0] * batch_size)
    sigma_b = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    dist_a = rl_distribs.Normal(mean=mu_a, scale=sigma_a)
    dist_b = rl_distribs.Normal(mean=mu_b, scale=sigma_b)
    kl = dist_a.kl(dist_b)
    expected_kl = ((mu_a - mu_b)**2 / (2 * sigma_b**2) + 0.5 * (
      (sigma_a**2 / sigma_b**2) - 1 - 2 * np.log(sigma_a / sigma_b)))
    self.assertArrayEqual(kl.shape, (batch_size,))
    self.assertAllClose(expected_kl, kl)
    # test estimate kl
    set_test_seed()
    draws = dist_a.sample(100000)
    sample_kl = dist_a.log_prob(draws) - dist_b.log_prob(draws)
    sample_kl = torch.mean(sample_kl, dim=0)
    self.assertAllClose(expected_kl, sample_kl, atol=0.0, rtol=1e-2)