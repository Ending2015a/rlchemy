# --- built in ---
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib.data import buffers as rl_buffers
from rlchemy.lib.data import samplers as rl_samplers
from test.utils import TestCase

class TestDataSamplersModule(TestCase):
  """Test rlchemy.lib.data.samplers module"""
  def test_uniform_sampler_with_base_buffer(self):
    capacity = 10
    batch = 1
    n_samples = 15
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    samp = rl_samplers.UniformSampler(buf)
    for i in range(n_samples):
      buf.add({'a': ([i], [i+1])})
      if i < capacity-1:
        self.assertFalse(buf.isfull)
        self.assertEqual(i+1, len(buf))
      else:
        self.assertTrue(buf.isfull)
        self.assertEqual(capacity, len(buf))
    exp = np.arange(n_samples-capacity, n_samples)
    exp_a0 = np.roll(exp, n_samples % capacity)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(buf.data['a'][0], exp_a0)
    self.assertArrayEqual(buf.data['a'][1], exp_a1)
    # test sample (batch=None)
    batch = samp()
    self.assertArrayEqual((capacity,), batch['a'][0].shape)
    self.assertArrayEqual((capacity,), batch['a'][1].shape)
    self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
    self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
    # test sample (batch=3)
    batch_size = 3
    batch = samp(batch_size=batch_size)
    self.assertArrayEqual((batch_size,), batch['a'][0].shape)
    self.assertArrayEqual((batch_size,), batch['a'][1].shape)
    self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
    self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
    # test sample (batch=3, seq=2)
    batch_size = 3
    seq_len = 2
    batch = samp(batch_size=batch_size, seq_len=seq_len)
    self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
    self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
    self.assertArrayEqual(batch['a'][0], buf[samp.indices]['a'][0])
    self.assertArrayEqual(batch['a'][1], buf[samp.indices]['a'][1])
    # test update
    batch['a'] = (np.zeros_like(batch['a'][0]), 
            np.zeros_like(batch['a'][1]))
    samp.update(batch)
    self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
    self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])


  def test_uniform_sampler_with_base_buffer_rel(self):
    capacity = 10
    batch = 1
    n_samples = 15
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    samp = rl_samplers.UniformSampler(buf)
    for i in range(n_samples):
      buf.add({'a': ([i], [i+1])})
    # test relative indexing
    inds1 = np.arange(3)
    inds2 = np.zeros(3, dtype=np.int64)
    samp._cached_inds = (inds1, inds2)
    self.assertArrayEqual(buf[samp.indices]['a'][0], samp.rel[0]['a'][0])
    self.assertArrayEqual(buf[samp.indices]['a'][1], samp.rel[0]['a'][1])
    self.assertArrayEqual(buf[(inds1-3, inds2)]['a'][0], samp.rel[-3]['a'][0])
    self.assertArrayEqual(buf[(inds1-3, inds2)]['a'][1], samp.rel[-3]['a'][1])
    self.assertArrayEqual(buf[(inds1+3, inds2)]['a'][0], samp.rel[3]['a'][0])
    self.assertArrayEqual(buf[(inds1+3, inds2)]['a'][1], samp.rel[3]['a'][1])
    add = np.array([[1, 2]], dtype=np.int64).T
    self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][0], samp.rel[1:3]['a'][0])
    self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][1], samp.rel[1:3]['a'][1])
    add = np.array([[-3, -2]], dtype=np.int64).T
    self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][0], samp.rel[-3:-1]['a'][0])
    self.assertArrayEqual(buf[(inds1+add, inds2)]['a'][1], samp.rel[-3:-1]['a'][1])
    # test shape
    self.assertArrayEqual((0, 3), samp.rel[-1:1]['a'][0].shape)
    # test setitem
    add = np.array([[1, 2]], dtype=np.int64).T
    batch = samp.rel[1:3]
    batch['a'] = (np.zeros_like(batch['a'][0]),
            np.zeros_like(batch['a'][1]))
    samp.rel[1:3] = batch
    self.assertTrue(np.all(samp.rel[1:3]['a'][0] == 0))
    self.assertTrue(np.all(samp.rel[1:3]['a'][1] == 0))

  def test_permute_sampler_with_base_buffer(self):
    capacity = 10
    batch = 1
    n_samples = 15
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    samp = rl_samplers.PermuteSampler(buf)
    for i in range(n_samples):
      buf.add({'a': ([i], [i+1])})
    # test sample (batch=None)
    batch_size = len(buf)
    batches = []
    indices = []
    for batch in samp():
      self.assertArrayEqual((batch_size,), batch['a'][0].shape)
      self.assertArrayEqual((batch_size,), batch['a'][1].shape)
      self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
      self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices, 
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(1, len(batches))
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once
    self.assertTrue(np.all(counts == 1))
    # test sample (batch=3)
    batch_size = 3
    batches = []
    indices = []
    for batch in samp(batch_size=batch_size):
      self.assertArrayEqual((batch_size,), batch['a'][0].shape)
      self.assertArrayEqual((batch_size,), batch['a'][1].shape)
      self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
      self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices, 
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(4, len(batches)) # total samples == capacity
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once but less than 2
    self.assertTrue(np.all(counts >= 1))
    self.assertTrue(np.all(counts <= 2))
    # test sample (batch=3, seq_len=2)
    batch_size = 3
    seq_len = 2
    batches = []
    indices = []
    for batch in samp(batch_size=batch_size, seq_len=seq_len):
      self.assertArrayEqual((batch_size, seq_len), batch['a'][0].shape)
      self.assertArrayEqual((batch_size, seq_len), batch['a'][1].shape)
      self.assertArrayEqual(buf[samp.indices]['a'][0], batch['a'][0])
      self.assertArrayEqual(buf[samp.indices]['a'][1], batch['a'][1])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices, 
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(3, len(batches)) # total samples == capacity
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once but less than 3
    self.assertTrue(np.all(counts >= 1))
    self.assertTrue(np.all(counts <= 4))
    
  def test_uniform_sampler_with_dynamic_buffer(self):
    n_samples = 10
    batch_ = 2
    buf = rl_buffers.DynamicBuffer(batch_)
    samp = rl_samplers.UniformSampler(buf)
    for i in range(n_samples):
      samp.add(a=[i, i+n_samples])
    self.assertEqual(n_samples*2, len(buf))
    self.assertFalse(buf.ready_for_sample)
    self.assertFalse(buf.isfull)
    # test sample (batch=None)
    with self.assertRaises(RuntimeError):
      # buffer is not ready for sampling
      samp()
    buf.make()
    # test sample (batch=None)
    batch = samp()
    self.assertArrayEqual((n_samples*batch_,), batch['a'].shape)
    self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
    # test sample (batch=3)
    batch_size = 3
    batch = samp(batch_size=batch_size)
    self.assertArrayEqual((batch_size,), batch['a'].shape)
    self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
    # test sample (batch=3, seq=2)
    batch_size = 3
    seq_len = 2
    batch = samp(batch_size=batch_size, seq_len=seq_len)
    self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
    self.assertArrayEqual(batch['a'], buf[samp.indices]['a'])
    # test update
    batch['a'] = np.zeros_like(batch['a'])
    samp.update(a=batch['a'])
    self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])

  def test_permute_sampler_with_dynamic_buffer(self):
    n_samples = 10
    batch_ = 2
    buf = rl_buffers.DynamicBuffer(batch_)
    samp = rl_samplers.PermuteSampler(buf)
    for i in range(n_samples):
      buf.add(a=[i, i+n_samples])
    self.assertEqual(n_samples*2, len(buf))
    self.assertFalse(buf.ready_for_sample)
    self.assertFalse(buf.isfull)
    # test sample (batch=None)
    with self.assertRaises(RuntimeError):
      # buffer is not ready for sampling
      samp()
    buf.make()
    # test sample (batch=None)
    batch_size = len(buf)
    batches = []
    indices = []
    rl_utils.set_seed(2)
    for batch in samp():
      self.assertArrayEqual((batch_size,), batch['a'].shape)
      self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices,
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(1, len(batches))
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once
    self.assertTrue(np.all(counts == 1))
    # test sample (batch=3)
    batch_size = 3
    batches = []
    indices = []
    for batch in samp(batch_size=batch_size):
      self.assertArrayEqual((batch_size,), batch['a'].shape)
      self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices, 
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(7, len(batches)) # total samples == capacity
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once but less than 2
    self.assertTrue(np.all(counts >= 1))
    self.assertTrue(np.all(counts <= 2))
    # test sample (batch=3, seq_len=2)
    batch_size = 3
    seq_len = 2
    batches = []
    indices = []
    rl_utils.set_seed(2)
    for batch in samp(batch_size=batch_size, seq_len=seq_len):
      self.assertArrayEqual((batch_size, seq_len), batch['a'].shape)
      self.assertArrayEqual(buf[samp.indices]['a'], batch['a'])
      batches.append(batch)
      indices.append(
        np.ravel_multi_index(samp.indices, 
              (buf.len_slots(), buf.batch))
      )
    self.assertEqual(6, len(batches)) # total samples == capacity
    unique, counts = np.unique(indices, return_counts=True)
    # check if contains all elements
    self.assertTrue(len(buf), len(unique))
    # check if all elements are sampled at least once but less than 3
    self.assertTrue(np.all(counts >= 1))
    self.assertTrue(np.all(counts <= 3))

  def test_prioritized_sampler_with_replay_buffer(self):
    capacity = 10
    batch = 1
    alpha = 1.0
    # test create space
    buf = rl_buffers.ReplayBuffer(capacity)
    self.assertTrue(buf.isnull)
    samp = rl_samplers.PrioritizedSampler(buf, alpha)
    self.assertTrue(samp._weight_tree is None)
    # add one sample to create space
    samp.add(i=[1])
    self.assertTrue(samp._weight_tree is not None)
    self.assertEqual(buf.capacity, samp._weight_tree._size)
    self.assertTrue(samp._weight_tree._base > buf.capacity)
    self.assertEqual(1, samp._weight_tree.sum())
    samp.add(i=[2])
    samp.add(i=[3])
    self.assertEqual(3, samp._weight_tree.sum())
    # test sample (batch=None)
    batch = samp.sample(beta=-1.0)
    self.assertEqual((3,), batch['i'].shape)
    self.assertArrayEqual(np.ones((3,), dtype=np.float32), batch['w'])
    # test sample (batch=2)
    rl_utils.set_seed(1) # i=[2, 3]
    batch_size = 2
    batch = samp.sample(batch_size=batch_size, beta=-1.0)
    self.assertArrayEqual([1, 2], samp.indices[0])
    self.assertEqual((2,), batch['i'].shape)
    self.assertArrayEqual(np.ones((2,), dtype=np.float32), batch['w'])
    samp.update(w=[0.5, 0.5])
    self.assertAllClose(0.5, samp._min_w) # exponent
    self.assertAllClose(1.0, samp._max_w) # exponent
    self.assertAllClose(2, samp._weight_tree.sum())
    batches = []
    for n in range(10000):
      batch = samp.sample(batch_size=batch_size, beta=-1.0) # i=[1, 1]
      batches.append(batch['i'])
    samples = np.asarray(batches).flatten()
    self.assertAllClose(0.5, np.sum(samples==1)/(n*2), atol=1e-2)
    self.assertAllClose(0.25, np.sum(samples==2)/(n*2), atol=1e-2)
    self.assertAllClose(0.25, np.sum(samples==3)/(n*2), atol=1e-2)
    # test sample (batch=3, seq=2)
    rl_utils.set_seed(2) # i=[[1, 2], [1, 2], [2, 3]]
    batch_size = 3
    seq_len = 2
    batch = samp.sample(batch_size=batch_size, seq_len=seq_len, beta=-1.0)
    self.assertArrayEqual([[0, 1], [0, 1], [1, 2]], samp.indices[0])
    self.assertEqual((3, 2), batch['i'].shape)
    self.assertAllClose([[1, 0.5], [1, 0.5], [0.5, 0.5]], batch['w'], atol=1e-6)
    samp.update(w=[[0.1, 0.2], [1, 0.5], [0.4, 0.5]])
    self.assertAllClose(0.1, samp._min_w)
    self.assertAllClose(1.0, samp._max_w)
    self.assertAllClose(1.9, samp._weight_tree.sum())

  def test_sampler_exception(self):
    with self.assertRaises(ValueError):
      # `buffer` must be an instance of BaseBuffer
      rl_samplers.UniformSampler(None)
    with self.assertRaises(ValueError):
      # `buffer` must be an instance of BaseBuffer
      rl_samplers.PermuteSampler(None)
    # with self.assertRaises(ValueError):
    #     # `buffer` must be ReplayBuffer
    #     rl_samplers.PrioritizedSampler(rl_buffers.BaseBuffer(10), 1.0)
    with self.assertRaises(ValueError):
      # `buffer` must be ReplayBuffer
      rl_samplers.PrioritizedSampler(rl_buffers.DynamicBuffer(), 1.0)
