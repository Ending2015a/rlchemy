# --- built in ---
import os
import sys
import time
import json
import logging
import tempfile
import unittest

# --- 3rd party ---
import gym
import numpy as np
import torch
from torch import nn

# --- my module ---
from rlchemy.lib.utils import common as rl_common
from test.utils import TestCase

def to_json_from_json(obj):
  serialized_obj = rl_common.to_json_serializable(obj)
  obj = rl_common.from_json_serializable(serialized_obj)
  return obj

def safe_json_dumps_loads(obj):
  string = rl_common.safe_json_dumps(obj)
  obj = rl_common.safe_json_loads(string)
  return obj


class TestUtilsCommonModule(TestCase):
  """Test rlchemy.lib.utils.common module"""
  def test_flatten_dicts(self):
    d1 = {'a': 1, 'b': [2, 3]}
    d2 = {'b': [4], 'a': 5}
    d3 = {'c': 6}
    d = rl_common.flatten_dicts([d1, d2, d3])
    self.assertEqual(list(d.keys()), ['a', 'b', 'c'])
    self.assertEqual(d['a'], [1, 5])
    self.assertEqual(d['b'], [[2, 3], [4]])
    self.assertEqual(d['c'], [6])

  def test_is_image_space(self):
    # Box 3D + uint8: True
    space = gym.spaces.Box(low=np.zeros((64,64,3)),
                 high=np.ones((64,64,3))*255, 
                 dtype=np.uint8)
    self.assertTrue(rl_common.is_image_space(space))
    # Box 2D + uint8: False
    space = gym.spaces.Box(low=np.zeros((64,64)),
                 high=np.ones((64,64))*255, 
                 dtype=np.uint8)
    self.assertFalse(rl_common.is_image_space(space))
    # Box 3D + float32: False
    space = gym.spaces.Box(low=np.zeros((64,64,3)),
                 high=np.ones((64,64,3)),
                 dtype=np.float32)
    self.assertFalse(rl_common.is_image_space(space))

  def test_is_bounded(self):
    # bounded non-box
    space = gym.spaces.Discrete(5)
    self.assertTrue(rl_common.is_bounded(space))
    # bounded box
    low = np.zeros((64,64,3))
    high = np.ones((64,64,3))
    space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    self.assertTrue(rl_common.is_bounded(space))
    # unbounded box: low >= high
    low = np.zeros((64,64,3))
    high = np.ones((64,64,3))
    high[0,0,0] = 0
    space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    self.assertFalse(rl_common.is_bounded(space))
    # unbounded box: inf
    low = np.zeros((64,64,3))
    high = np.ones((64,64,3))
    high[0,0,0] = np.inf
    space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    self.assertFalse(rl_common.is_bounded(space))

  def test_json_serializable_simple(self):
    """Test
      utils.common.to_json_serializable
      utils.common.from_json_serializable

      Test object:
        int, float, dict, tuple
    """
    # json serializable
    a = dict(a=10, b=1.0, c=[4.0, 5.0],
        d=('df', 'as'))

    self.assertEqual(rl_common.to_json_serializable(a), a)
    self.assertEqual(to_json_from_json(a), a)

    # not json serializable
    a[(8, 9, 10)] = 'asdf'
    self.assertEqual(to_json_from_json(a), a)

  def test_json_serializable_complex(self):
    """Test
      utils.common.to_json_serializable
      utils.common.from_json_serializab.e

      Test object:
        class, numpy
    """

    class A:
      c = 20
      def __init__(self, a, b):
        self.a = a
        self.b = b

    # class
    a = A(10, 11)
    a2 = to_json_from_json(a)
    self.assertEqual(type(a), type(a2))
    self.assertEqual(a.a, a2.a)
    self.assertEqual(a.b, a2.b)
    self.assertEqual(A.c, a2.c)

    # numpy
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = to_json_from_json(a)
    self.assertTrue(np.array_equal(a, a2))
    self.assertEqual(a.dtype, a2.dtype)

  def test_safe_json_dumps_loads_simple(self):
    # json serializable
    a = dict(a=10, b=1.0, c=[4.0, 5.0],
        d=('df', 'as'))

    self.assertEqual(rl_common.safe_json_dumps(a), 
            json.dumps(a,indent=2,ensure_ascii=False))
    self.assertEqual((a), a)

  def test_safe_json_dumps_loads_complex(self):

    class A:
      c = 20
      def __init__(self, a, b):
        self.a = a
        self.b = b

    # class
    a = A(10, 11)
    a2 = safe_json_dumps_loads(a)
    self.assertEqual(type(a), type(a2))
    self.assertEqual(a.a, a2.a)
    self.assertEqual(a.b, a2.b)
    self.assertEqual(A.c, a2.c)

    # class in dict
    a = {'a': 10, 'b': A(10, 11)}
    a2 = json.loads(rl_common.safe_json_dumps(a))
    self.assertEqual(a['a'], a2['a'])
    self.assertTrue(rl_common.is_json_serializable(a2))
    self.assertTrue(isinstance(a2['b'], dict))
    self.assertTrue(rl_common.SERIALIZATION_KEY in a2['b'].keys())

    # numpy
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    a2 = safe_json_dumps_loads(a)
    self.assertTrue(np.array_equal(a, a2))
    self.assertEqual(a.dtype, a2.dtype)

  def test_state_object(self):

    # test state object
    s = rl_common.StateObject(a=10, b=20)
    s2 = rl_common.StateObject.fromstring(s.tostring())
    self.assertEqual(s2, s)

    # test assign values to state object
    s = rl_common.StateObject(a=np.array([1, 2, 3], dtype=np.uint8))
    s.b = 20
    s.c = None
    s['d'] = 30
    s2 = rl_common.StateObject.fromstring(s.tostring())

    self.assertEqual(s2['b'], 20)
    self.assertEqual(s2['c'], None)
    self.assertEqual(s2.d, 30)
    self.assertEqual(s.keys(), s2.keys())
    for k, v in s.items():
      if isinstance(v, np.ndarray):
        self.assertTrue(np.array_equal(s[k], s2[k]))
      else:
        self.assertEqual(s[k], s2[k])