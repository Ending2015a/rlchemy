# --- built in ---
# --- 3rd party ---
import numpy as np
# --- my module ---
from rlchemy.lib.data import utils as data_utils
from test.utils import TestCase

class TestDataUtilsModule(TestCase):
    """Test rlchemy.lib.data.utils module"""
    def test_segment_tree(self):
        tree = data_utils.SegmentTree(10)
        # test getitem/setitem
        tree[:] = np.arange(1, 11)
        self.assertArrayEqual(np.arange(1, 11), tree[:])
        tree[np.arange(3)] = [3, 2, 1]
        self.assertArrayEqual([3, 2, 1, 4], tree[np.arange(4)])
        # test sum
        tree[:] = np.arange(1, 11)
        self.assertEqual(55, tree.sum())
        self.assertEqual(5, tree.sum(start=1, end=3))
        self.assertEqual(7, tree.sum(start=2, end=4))
        self.assertEqual(44, tree.sum(start=1, end=-1))
        # test update
        tree.update(1, 3)
        self.assertEqual(6, tree.sum(start=1, end=3))
        # test index
        tree[:] = np.arange(1, 11)
        self.assertEqual(0, tree.index(1))
        self.assertEqual(1, tree.index(2))
        self.assertEqual(1, tree.index(3))
        self.assertEqual(2, tree.index(4))
        self.assertEqual(3, tree.index(7))
        self.assertEqual(8, tree.index(45))
        self.assertEqual(9, tree.index(54))
        # test index vectorized
        self.assertArrayEqual([[0], [9]], tree.index([[1], [46]]))

    def test_compute_nstep_rew(self):
        rew = np.asarray([[1., 0., 1.], [1., 1., 1.]], dtype=np.float32).T
        done = np.asarray([[0., 0., 0.], [0., 1., 0.]], dtype=np.float32).T
        gamma = 0.95
        res = data_utils.compute_nstep_rew(rew, done, gamma=gamma)
        exp = np.asarray([1.9025, 1.95], dtype=np.float32)
        self.assertAllClose(exp, res)

    def test_compute_nstep_rew_empty(self):
        rew = np.asarray([], dtype=np.float32)
        done = np.asarray([], dtype=np.float32)
        res = data_utils.compute_nstep_rew(rew, done, 0.95)
        self.assertArrayEqual(rew, res)

    def test_compute_advantage(self):
        rew = np.asarray([0, 0, 1, 0, 1], dtype=np.float32)
        val = np.asarray([.0, .1, .5, .4, .5], dtype=np.float32)
        done = np.asarray([0, 0, 1, 0, 0], dtype=np.bool_)
        adv = data_utils.compute_advantage(rew=rew, val=val, done=done, 
                                    gamma=0.99, gae_lambda=0.95)
        # openai baselines style GAE
        exp = np.asarray([0.00495, -0.1, 1.865465, 1.0307975, 0.995],
                        dtype=np.float32)
        # tianshou style GAE:
        # exp = np.asarray([1.26304564, 1.23768806, 0.89600003, 0.56525001, 0.5],
        #                 dtype=np.float32)
        self.assertAllClose(exp, adv)