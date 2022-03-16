# --- built in ---
# --- 3rd party ---
import numpy as np
import torch
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

def make_categorical(batch_shape, num_classes, dtype=torch.int32, seed=get_test_seed()):
  rl_utils.set_seed(seed)
  shape = list(batch_shape)+[num_classes]
  logits = -50. + torch.zeros(shape).uniform_(-10, 10)
  return rl_distribs.Categorical(logits.float(), dtype=dtype)

class TestDistribsModuleCategorical(TestCase):
    @parameterized.expand([
        ([],),
        ([1],),
        ([2, 3, 4],)
    ])
    def test_shapes(self, batch_shape):
        shape = list(batch_shape) + [10]
        dist = make_categorical(batch_shape, 10, dtype=torch.int32)
        self.assertEqual(1, dist.event_ndims)
        self.assertArrayEqual(shape, dist.shape)
        self.assertArrayEqual(batch_shape, dist.batch_shape)
        self.assertArrayEqual([10], dist.event_shape)
        self.assertArrayEqual(shape, dist.logits.shape)
        self.assertArrayEqual(shape, dist.probs().shape)
        self.assertArrayEqual(batch_shape, dist.log_prob(np.zeros(batch_shape)).shape)
        self.assertArrayEqual(batch_shape, dist.mode().shape)
        self.assertArrayEqual(batch_shape, dist.sample().shape)
        self.assertArrayEqual(batch_shape, dist.entropy().shape)
        dist2 = make_categorical(batch_shape, 10, dtype=torch.int32)
        self.assertArrayEqual(batch_shape, dist.kl(dist2).shape)
    
    @parameterized.expand([
        (torch.int32,),
        (torch.int64,),
    ])
    def test_dtypes(self, dtype):
        dist = make_categorical([], 10, dtype=dtype)
        self.assertEqual(dtype, dist.dtype)
        self.assertEqual(torch.float32, dist.probs().dtype)
        self.assertEqual(torch.float32, dist.log_prob(0).dtype)
        self.assertEqual(dtype, dist.mode().dtype)
        self.assertEqual(dtype, dist.sample().dtype)
        self.assertEqual(torch.float32, dist.entropy().dtype)
        dist2 = make_categorical([], 10, dtype=dtype)
        self.assertEqual(torch.float32, dist.kl(dist2).dtype)

    def test_prob_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test logits
        self.assertArrayEqual(dist.logits, logits)
        self.assertArrayEqual(dist.logits.shape, p.shape)
        # test _p, _log_p
        self.assertAllClose(dist.probs(), p)
        # test prob, log_prob
        self.assertAllClose(dist.prob(0), p[0])
        self.assertAllClose(dist.prob(1), p[1])

    def test_mode_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test mode
        modes = dist.mode()
        self.assertArrayEqual(1, modes)

    def test_sample_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test sample
        set_test_seed()
        draws = rl_utils.to_numpy(dist.sample(10000))
        self.assertFalse(np.any(draws < 0))
        self.assertFalse(np.any(draws > 1))
        self.assertAllClose(np.mean(draws==0, axis=0), 0.2, atol=1e-2)
        self.assertAllClose(np.mean(draws==1, axis=0), 0.8, atol=1e-2)

    def test_entropy_without_batch(self):
        p = np.array([0.2, 0.8], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test entropy
        self.assertAllClose(-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                        dist.entropy(), atol=0, rtol=1e-5)
    
    def test_prob_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test logits
        self.assertArrayEqual(dist.logits, logits)
        self.assertArrayEqual(dist.logits.shape, p.shape)
        # test _p, _log_p
        self.assertAllClose(dist.probs(), p)
        # test prob, log_prob
        self.assertAllClose(dist.prob([0, 0]), p[..., 0])
        self.assertAllClose(dist.prob([1, 1]), p[..., 1])

    def test_mode_with_batch(self):
        p = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test mode
        modes = dist.mode()
        self.assertArrayEqual([1, 0], modes)

    def test_sample_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        # test sample
        set_test_seed()
        draws = rl_utils.to_numpy(dist.sample(10000))
        self.assertFalse(np.any(draws < 0))
        self.assertFalse(np.any(draws > 1))
        self.assertAllClose(np.mean(draws==0, axis=0), [0.2, 0.4], atol=1e-2)
        self.assertAllClose(np.mean(draws==1, axis=0), [0.8, 0.6], atol=1e-2)
    
    def test_entropy_with_batch(self):
        p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float32)
        logits = rl_utils.to_tensor(np.log(p) - 50.)
        dist = rl_distribs.Categorical(logits)
        self.assertAllClose([-(0.2 * np.log(0.2) + 0.8 * np.log(0.8)),
                             -(0.4 * np.log(0.4) + 0.6 * np.log(0.6))],
                             dist.entropy(), atol=0, rtol=1e-5)
    
    def test_entropy_with_neg_inf_logits(self):
        probs = np.array([[0, 0.5, 0.5], [0, 1, 0]])
        dist = rl_distribs.Categorical(np.log(probs))

        ans = [-(0.5*np.log(0.5) + 0.5*np.log(0.5)), -(np.log(1))]
        self.assertAllClose(ans, dist.entropy())

    @parameterized.expand([
        ([1], 2),
        ([1], 4),
        ([10], 2),
        ([10], 4)
    ])
    def test_categorical_kl(self, batch_shape, num_classes):
        dist_a = make_categorical(batch_shape, num_classes, seed=1)
        dist_b = make_categorical(batch_shape, num_classes, seed=2)

        exp_a = np.exp(rl_utils.to_numpy(dist_a.logits))
        exp_b = np.exp(rl_utils.to_numpy(dist_b.logits))
        prob_a = exp_a / exp_a.sum(axis=-1, keepdims=True)
        prob_b = exp_b / exp_b.sum(axis=-1, keepdims=True)

        kl_val = dist_a.kl(dist_b)
        kl_same = dist_a.kl(dist_a)
        kl_exp = np.sum(prob_a * (np.log(prob_a) - np.log(prob_b)), axis=-1)
        self.assertArrayEqual(kl_val.shape, batch_shape)
        self.assertAllClose(kl_val, kl_exp, rtol=1e-6, atol=1e-6)
        self.assertAllClose(kl_same, np.zeros_like(kl_exp), rtol=1e-6, atol=1e-6)