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
from rlchemy.lib.utils import nest as rl_nest
from test.utils import TestCase

class TestUtilsNestModule(TestCase):
    """Test rlchemy.lib.utils.nest module"""

    def test_iter_nested(self):
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'b': (d1, d2), 'a': d3}
        res = list(rl_nest.iter_nested(data))
        self.assertEqual(3, len(res))
        self.assertArrayEqual(res[0], d1)
        self.assertArrayEqual(res[1], d2)
        self.assertArrayEqual(res[2], d3)
        # sort key
        res = list(rl_nest.iter_nested(data, sortkey=True))
        self.assertEqual(3, len(res))
        self.assertArrayEqual(res[0], d3)
        self.assertArrayEqual(res[1], d1)
        self.assertArrayEqual(res[2], d2)

    def test_map_nested(self):
        op = lambda v: len(v)
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        res = rl_nest.map_nested(data, op)
        self.assertEqual(res['a'][0], 3)
        self.assertEqual(res['a'][1], 4)
        self.assertEqual(res['b'], 5)

    def test_map_nested_exception(self):
        # ValueError when op is not a callable type
        op = True
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        with self.assertRaises(ValueError):
            rl_nest.map_nested(data, op)

    def test_iter_nested_space(self):
        sp1 = gym.spaces.Box(low=-1, high=1, shape=(16,),
                            dtype=np.float32)
        sp2 = gym.spaces.Discrete(6)
        sp = gym.spaces.Dict({'b': sp1, 'a': sp2})
        sp = gym.spaces.Tuple((sp,))
        # gym.spaces.Dict sorts keys when constructing.
        spaces = list(rl_nest.iter_nested_space(sp))
        self.assertEqual(2, len(spaces))
        self.assertEqual(sp2, spaces[0])
        self.assertEqual(sp1, spaces[1])
        # sort key
        spaces = list(rl_nest.iter_nested_space(sp, sortkey=True))
        self.assertEqual(2, len(spaces))
        self.assertEqual(sp2, spaces[0])
        self.assertEqual(sp1, spaces[1])

    def test_map_nested_space(self):
        op = lambda space: space.shape
        sp1 = gym.spaces.Box(low=-1, high=1, shape=(16,),
                            dtype=np.float32)
        sp2 = gym.spaces.Discrete(6)
        sp = gym.spaces.Dict({'b': sp1, 'a': sp2})
        sp = gym.spaces.Tuple((sp,))
        res = rl_nest.map_nested_space(sp, op)
        self.assertEqual(sp2.shape, res[0]['a'])
        self.assertEqual(sp1.shape, res[0]['b'])

    def test_iter_nested_tuple(self):
        data1 = {'a': (1, 2), 'b': 3}
        data2 = {'a': (4, 5), 'b': 6}
        res = list(rl_nest.iter_nested_tuple((data1, data2)))
        self.assertEqual(3, len(res))
        self.assertEqual((1, 4), res[0])
        self.assertEqual((2, 5), res[1])
        self.assertEqual((3, 6), res[2])

    def test_iter_nested_tuple_exception(self):
        data1 = {'a': (1, 2), 'b': 3}
        data2 = {'a': (4, 5), 'b': 6}
        with self.assertRaises(TypeError):
            res = rl_nest.iter_nested_tuple([data1, data2])

    def test_map_nested_tuple(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data1 = {'a': (d1, d2), 'b': d3}
        data2 = {'a': (d1, d2), 'b': d3}
        res = rl_nest.map_nested_tuple((data1, data2), op)
        self.assertEqual(res['a'][0], (2, 3))
        self.assertEqual(res['a'][1], (2, 4))
        self.assertEqual(res['b'], (2, 5))

    def test_map_nested_tuple_exception(self):
        op = lambda data_tuple: np.asarray(data_tuple).shape
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data1 = {'a': (d1, d2), 'b': d3}
        data2 = {'a': (d1, d2), 'b': d3}
        with self.assertRaises(TypeError):
            res = rl_nest.map_nested_tuple([data1, data2], op)
        with self.assertRaises(ValueError):
            op = None
            res = rl_nest.map_nested_tuple((data1, data2), op)

    def test_nested_to_numpy(self):
        d1 = list(range(3))
        d2 = list(range(4))
        d3 = list(range(5))
        data = {'a': (d1, d2), 'b': d3}
        
        res = rl_nest.nested_to_numpy(data)
        self.assertArrayEqual(res['a'][0], np.arange(3))
        self.assertArrayEqual(res['a'][1], np.arange(4))
        self.assertArrayEqual(res['b'], np.arange(5))

    def test_unpack_structure(self):
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        struct, flat_data = rl_nest.unpack_structure(data)

        self.assertEqual(len(flat_data), 3)
        self.assertArrayEqual(flat_data[0], d1)
        self.assertArrayEqual(flat_data[1], d2)
        self.assertArrayEqual(flat_data[2], d3)
        self.assertEqual(list(struct.keys()), list(data.keys()))
        self.assertTrue(isinstance(struct['a'], tuple))
        self.assertEqual(struct['a'][0], 0)
        self.assertEqual(struct['a'][1], 1)
        self.assertEqual(struct['b'], 2)

    def test_pack_sequence(self):
        d1 = np.arange(3)
        d2 = np.arange(4)
        d3 = np.arange(5)
        data = {'a': (d1, d2), 'b': d3}
        struct, flat_data = rl_nest.unpack_structure(data)
        res_data = rl_nest.pack_sequence(struct, flat_data)
        self.assertEqual(list(data.keys()), list(res_data.keys()))
        self.assertTrue(isinstance(res_data['a'], tuple))
        self.assertArrayEqual(res_data['a'][0], d1)
        self.assertArrayEqual(res_data['a'][1], d2)
        self.assertArrayEqual(res_data['b'], d3)
    
    def test_flatten_space(self):
        space = gym.spaces.Discrete(6)
        res = rl_nest.flatten_space(space)
        self.assertEqual(1, len(res))
        self.assertEqual(space, res[0])

        sp1 = gym.spaces.Discrete(6)
        sp2 = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        space = gym.spaces.Tuple((sp1, sp2))
        res = rl_nest.flatten_space(space)
        self.assertEqual(2, len(res))
        self.assertEqual(sp1, space[0])
        self.assertEqual(sp2, space[1])

        sp3 = gym.spaces.MultiDiscrete([3, 4])
        sp = gym.spaces.Tuple((sp2, sp3))
        space = gym.spaces.Dict({'a': sp1, 'b': sp})
        res = rl_nest.flatten_space(space)
        self.assertEqual(3, len(res))
        self.assertEqual(sp1, res[0])
        self.assertEqual(sp2, res[1])
        self.assertEqual(sp3, res[2])