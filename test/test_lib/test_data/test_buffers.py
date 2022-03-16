# --- built in ---
import math
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib.data import buffers as rl_buffers
from test.utils import TestCase

class TestDataBuffersModule(TestCase):
  """Test rlchemy.lib.data.buffers module"""
  def test_base_buffer(self):
    capacity = 10
    batch = 1
    n_samples = 15 # test circular
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    self.assertEqual(capacity, buf.capacity)
    self.assertEqual(capacity, buf.slots)
    self.assertEqual(batch, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertTrue(buf.ready_for_sample)
    for i in range(n_samples):
      buf.add({'a': ([i], [i+1])})
      if i < capacity-1:
        self.assertFalse(buf.isfull)
        self.assertEqual(i+1, len(buf))
        self.assertEqual(i+1, buf.len_slots())
        self.assertEqual(0, buf.head)
      else:
        self.assertTrue(buf.isfull)
        self.assertEqual(capacity, len(buf))
        self.assertEqual(capacity, buf.len_slots())
      self.assertEqual((i+1)%capacity, buf.tail)
    exp = np.arange(n_samples-capacity, n_samples)
    exp_a0 = np.roll(exp, n_samples % capacity)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, buf.data['a'][0])
    self.assertArrayEqual(exp_a1, buf.data['a'][1])
    # test getitem
    data = buf[np.arange(n_samples % capacity)]
    exp_a0 = np.arange(n_samples - n_samples % capacity, n_samples)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, data['a'][0])
    self.assertArrayEqual(exp_a1, data['a'][1])
    # test setitem
    n = n_samples - capacity
    new_data = np.arange(n - n_samples % capacity, n)
    new_data = np.expand_dims(new_data, axis=-1)
    new_data = {'a': (new_data, new_data+1)}
    buf[np.arange(n_samples % capacity)] = new_data
    n = n_samples - capacity - n_samples % capacity
    exp_a0 = np.arange(n, n + capacity)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, buf.data['a'][0])
    self.assertArrayEqual(exp_a1, buf.data['a'][1])
    # test update (should have the same results as setitem)
    buf.update(new_data, indices=np.arange(n_samples % capacity))
    self.assertArrayEqual(exp_a0, buf.data['a'][0])
    self.assertArrayEqual(exp_a1, buf.data['a'][1])
    # test ravel/unravel index
    def test_ravel(indices):
      self.assertArrayEqual(
        np.ravel_multi_index(indices, (buf.slots, buf.batch)),
        buf.ravel_index(indices))
    test_ravel(([1, 2, 3], 0))
    test_ravel(([1, 2, 3], [0]))
    def test_unravel(indices):
      self.assertArrayEqual(
        np.unravel_index(indices, (buf.slots, buf.batch)),
        buf.unravel_index(indices))
    test_unravel([4, 5, 6])
    test_unravel(7)

  def test_base_buffer_multidim(self):
    capacity = 20
    batch = 2
    dim = 2
    n_samples = 15 # test circular
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    data = np.arange(n_samples*batch*dim).reshape((n_samples, batch, dim))
    for i in range(n_samples):
      buf.add({'a': data[i]})
      if (i+1)*batch < capacity:
        self.assertFalse(buf.isfull)
        self.assertEqual((i+1)*batch, len(buf))
        self.assertEqual(i+1, buf.len_slots())
        self.assertEqual(0, buf.head)
      else:
        self.assertTrue(buf.isfull)
        self.assertEqual(capacity, len(buf))
        self.assertEqual(capacity//batch, buf.len_slots())
      self.assertEqual((i+1)%(capacity//batch), buf.tail)
    exp = np.arange(n_samples*batch*dim-capacity*dim, n_samples*batch*dim)
    exp = exp.reshape(-1, 2, 2)
    exp = np.roll(exp, n_samples % (capacity//batch), axis=0)
    self.assertArrayEqual(exp, buf.data['a'])
    # test ravel/unravel index
    def test_ravel(indices):
      self.assertArrayEqual(
        np.ravel_multi_index(indices, (buf.slots, buf.batch)),
        buf.ravel_index(indices))
    test_ravel(([1, 2, 3], 0))
    test_ravel(([[1], [2], [3]], [0, 1]))
    def test_unravel(indices):
      self.assertArrayEqual(
        np.unravel_index(indices, (buf.slots, buf.batch)),
        buf.unravel_index(indices))
    test_unravel([4, 5, 6])
    test_unravel(7)

  def test_base_buffer_auto_calc_space(self):
    capacity = 10
    batch = 1
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    self.assertEqual(0, len(buf))
    self.assertEqual(0, buf.len_slots())
    self.assertEqual(capacity, buf.capacity)
    self.assertEqual(capacity, buf.slots)
    self.assertEqual(batch, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertTrue(buf.ready_for_sample)
    capacity = 10
    n_samples = 15 # test circular
    buf = rl_buffers.BaseBuffer(capacity, batch=None)
    self.assertEqual(0, len(buf))
    self.assertEqual(0, buf.len_slots())
    self.assertEqual(None, buf.capacity)
    self.assertEqual(None, buf.slots)
    self.assertEqual(None, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertTrue(buf.ready_for_sample)
    buf.add({'a': [0, 1]})
    self.assertEqual(2, len(buf))
    self.assertEqual(1, buf.len_slots())
    self.assertEqual(capacity, buf.capacity)
    self.assertEqual(math.ceil(capacity/2), buf.slots)
    self.assertEqual(2, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(1, buf.tail)
    self.assertFalse(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertTrue(buf.ready_for_sample)

  def test_base_buffer_relative_index(self):
    capacity = 10
    batch = 1
    n_samples = 15 # test circular
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    for i in range(n_samples):
      buf.add({'a': ([i], [i+1])})
    head = n_samples%capacity
    self.assertEqual(head, buf.head)
    self.assertEqual(head, buf.tail)
    # test int, slice key
    data = buf.rel[1]
    self.assertArrayEqual([head+1], data['a'][0])
    self.assertArrayEqual([head+2], data['a'][1])
    data = buf.rel[-1]
    self.assertArrayEqual([n_samples-1], data['a'][0])
    self.assertArrayEqual([n_samples], data['a'][1])
    data = buf.rel[1:3]
    exp = np.arange(2).reshape(-1, 1)
    self.assertArrayEqual(exp+head+1, data['a'][0])
    self.assertArrayEqual(exp+head+2, data['a'][1])
    data = buf.rel[-3:-1]
    exp = np.arange(2, 0, -1).reshape(-1, 1)
    self.assertArrayEqual(n_samples-1-exp, data['a'][0])
    self.assertArrayEqual(n_samples-exp, data['a'][1])
    data = buf.rel[-1:1]
    self.assertEqual((0, 1), data['a'][0].shape)
    self.assertEqual((0, 1), data['a'][1].shape)
    # test tuple key
    data = buf.rel[1, 0]
    self.assertArrayEqual(head+1, data['a'][0])
    self.assertArrayEqual(head+2, data['a'][1])
    data = buf.rel[-1, 0]
    self.assertArrayEqual(n_samples-1, data['a'][0])
    self.assertArrayEqual(n_samples, data['a'][1])
    data = buf.rel[1:3, 0]
    exp = np.arange(2)
    self.assertArrayEqual(exp+head+1, data['a'][0])
    self.assertArrayEqual(exp+head+2, data['a'][1])
    data = buf.rel[-3:-1, 0]
    exp = np.arange(2, 0, -1)
    self.assertArrayEqual(n_samples-1-exp, data['a'][0])
    self.assertArrayEqual(n_samples-exp, data['a'][1])
    data = buf.rel[-1:1, 0]
    self.assertEqual((0,), data['a'][0].shape)
    self.assertEqual((0,), data['a'][1].shape)
    # test list key
    data = buf.rel[[1,3]]
    self.assertArrayEqual([[head+1], [head+3]], data['a'][0])
    self.assertArrayEqual([[head+2], [head+4]], data['a'][1])
    # test np key
    data = buf.rel[np.asarray([1,3])]
    self.assertArrayEqual([[head+1], [head+3]], data['a'][0])
    self.assertArrayEqual([[head+2], [head+4]], data['a'][1])
    # test index out of range
    with self.assertRaises(IndexError):
      buf.rel[capacity+1]

  def test_base_buffer_shape(self):
    capacity = 10
    batch = 3
    n_samples = 15
    buf = rl_buffers.BaseBuffer(capacity, batch=batch)
    for i in range(n_samples):
      buf.add({'a': np.arange(batch)})
    self.assertArrayEqual((3,), buf[1]['a'].shape)
    self.assertArrayEqual((2,3), buf[1:3]['a'].shape)
    self.assertArrayEqual((2,), buf[1:3, 0]['a'].shape)
    self.assertArrayEqual((2,2), buf[1:3, :2]['a'].shape)
    self.assertArrayEqual((3,), buf.rel[1]['a'].shape)
    self.assertArrayEqual((2,3), buf.rel[1:3]['a'].shape)
    self.assertArrayEqual((2,), buf.rel[1:3, 0]['a'].shape)
    self.assertArrayEqual((2,2), buf.rel[1:3, :2]['a'].shape)

  def test_base_buffer_exception(self):
    with self.assertRaises(ValueError):
      # size <= 0
      rl_buffers.BaseBuffer(0, 1)
    with self.assertRaises(AssertionError):
      # batch <= 0
      rl_buffers.BaseBuffer(1, 0)
    buf = rl_buffers.BaseBuffer(1, 1)
    self.assertTrue(buf.isnull)
    with self.assertRaises(AssertionError):
      # AssertionError: Buffer space not created
      buf[0]
    # auto create space
    buf._set_data(1, indices=0)
    buf[0]

  def test_replay_buffer(self):
    capacity = 10
    batch = 1
    n_samples = 15 # test circular
    buf = rl_buffers.ReplayBuffer(capacity, batch=batch)
    self.assertEqual(capacity, buf.capacity)
    self.assertEqual(capacity, buf.slots)
    self.assertEqual(batch, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertTrue(buf.ready_for_sample)
    for i in range(n_samples):
      buf.add(a=[i], b=[i+1])
      if i < capacity-1:
        self.assertFalse(buf.isfull)
        self.assertEqual(i+1, len(buf))
        self.assertEqual(i+1, buf.len_slots())
        self.assertEqual(0, buf.head)
      else:
        self.assertTrue(buf.isfull)
        self.assertEqual(capacity, len(buf))
        self.assertEqual(capacity, buf.len_slots())
      self.assertEqual((i+1)%capacity, buf.tail)
    self.assertEqual(set(buf.keys()), set(['a', 'b']))
    self.assertFalse('c' in buf)
    exp = np.arange(n_samples-capacity, n_samples)
    exp_a0 = np.roll(exp, n_samples % capacity)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, buf.data['a'])
    self.assertArrayEqual(exp_a1, buf.data['b'])
    # test getitem
    data = buf[np.arange(n_samples % capacity)]
    exp_a0 = np.arange(n_samples - n_samples % capacity, n_samples)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, data['a'])
    self.assertArrayEqual(exp_a1, data['b'])
    # test setitem
    n = n_samples - capacity
    new_data = np.arange(n - n_samples % capacity, n)
    new_data = np.expand_dims(new_data, axis=-1)
    new_data = {'a': new_data, 'b': new_data+1}
    buf[np.arange(n_samples % capacity)] = new_data
    n = n_samples - capacity - n_samples % capacity
    exp_a0 = np.arange(n, n + capacity)
    exp_a1 = exp_a0 + 1
    exp_a0 = np.expand_dims(exp_a0, axis=-1)
    exp_a1 = np.expand_dims(exp_a1, axis=-1)
    self.assertArrayEqual(exp_a0, buf.data['a'])
    self.assertArrayEqual(exp_a1, buf.data['b'])
    # test update (should have the same results as setitem)
    buf.update(a=new_data['a'], b=new_data['b'], 
          indices=np.arange(n_samples % capacity))
    self.assertArrayEqual(exp_a0, buf.data['a'])
    self.assertArrayEqual(exp_a1, buf.data['b'])

  def test_replay_buffer_exception(self):
    buf = rl_buffers.ReplayBuffer(1, 1)
    self.assertEqual([], buf.keys())
    self.assertFalse('k' in buf)
    with self.assertRaises(AssertionError):
      buf.add(a=[1])
      buf.add(b=[2]) # key not exists

  def test_dynamic_buffer(self):
    n_samples = 10
    buf = rl_buffers.DynamicBuffer(batch=2)
    self.assertEqual(np.inf, buf.capacity)
    self.assertEqual(0, buf.head)
    for i in range(n_samples):
      buf.add(a=[i, i+n_samples])
    self.assertEqual(n_samples*2, len(buf))
    self.assertEqual(n_samples, buf.len_slots())
    self.assertEqual(n_samples, buf.slots)
    self.assertFalse(buf.ready_for_sample)
    self.assertFalse(buf.isfull)
    buf.make()
    self.assertEqual(n_samples*2, len(buf))
    self.assertTrue(buf.ready_for_sample)
    self.assertFalse(buf.isfull)
    exp = np.arange(n_samples*2).reshape(2, -1).T
    self.assertArrayEqual(exp, buf.data['a'])
    # test getitem
    data = buf[np.arange(n_samples//2)]
    exp = np.c_[np.arange(n_samples//2), np.arange(n_samples//2)+n_samples]
    self.assertArrayEqual(exp, data['a'])
    # test setitem
    new_data = exp+1
    buf[np.arange(n_samples//2)] = {'a':new_data}
    exp = np.c_[np.arange(n_samples), np.arange(n_samples)+n_samples]
    exp[:n_samples//2] += 1
    self.assertArrayEqual(exp, buf.data['a'])
    # test update (should have the same results as setitem)
    buf.update(a=new_data, indices=np.arange(n_samples//2))
    self.assertArrayEqual(exp, buf.data['a'])

  def test_dynamic_buffer_auto_calc_space(self):
    batch = 2
    buf = rl_buffers.DynamicBuffer(batch=batch)
    self.assertEqual(0, len(buf))
    self.assertEqual(0, buf.len_slots())
    self.assertEqual(np.inf, buf.capacity)
    self.assertEqual(0, buf.slots)
    self.assertEqual(batch, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertFalse(buf.ready_for_sample)
    # auto calc space
    buf = rl_buffers.DynamicBuffer(batch=None)
    self.assertEqual(0, len(buf))
    self.assertEqual(0, buf.len_slots())
    self.assertEqual(np.inf, buf.capacity)
    self.assertEqual(0, buf.slots)
    self.assertEqual(None, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(0, buf.tail)
    self.assertTrue(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertFalse(buf.ready_for_sample)
    buf.add(a=[0, 1])
    self.assertEqual(2, len(buf))
    self.assertEqual(1, buf.len_slots())
    self.assertEqual(np.inf, buf.capacity)
    self.assertEqual(1, buf.slots)
    self.assertEqual(2, buf.batch)
    self.assertEqual(0, buf.head)
    self.assertEqual(1, buf.tail)
    self.assertFalse(buf.isnull)
    self.assertFalse(buf.isfull)
    self.assertFalse(buf.ready_for_sample)

  def test_dynamic_buffer_exception(self):
    buf = rl_buffers.DynamicBuffer(batch=1)
    buf.add(a=[1])
    self.assertFalse(buf.ready_for_sample)
    with self.assertRaises(RuntimeError):
      # not ready for sample
      buf.update(indices=0, a=[2])
    buf.make()
    with self.assertRaises(RuntimeError):
      # make twice: "The buffer has already made."
      buf.make()
    with self.assertRaises(RuntimeError):
      # "The buffer can not add data after calling `buffer.make()`"
      buf.add(a=[2])
    with self.assertRaises(RuntimeError):
      buf._append_data({'a': [1]})
