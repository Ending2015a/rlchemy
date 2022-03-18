# --- bulid in ---
import os
import sys
import time
from typing import (
  Any,
  Tuple
)
# --- 3rd party ---
import gym
import numpy as np
from gym.wrappers import TimeLimit
# --- my module ---

__all__ = [
  'TimeLimit',
  'TimeFeatureWrapper'
]

# === Wrappers for Continuous Tasks ===

# Borrowed from Stable baselines
class TimeFeatureWrapper(gym.Wrapper):
  def __init__(
    self,
    env: gym.Env,
    max_steps: int = 1000,
    test_mode: bool = True
  ):
    """
    Add remaining time to observation space for fixed length episodes.
    See:
    * https://arxiv.org/abs/1712.00378
    * https://github.com/aravindr93/mjrl/issues/13.
    
    Args:
      env (gym.Env): environment to wrap.
      max_steps (int, optional): max number of steps of an episode if it is
        not wrapped in a TimeLimit object.
      test_mode (bool, optional): in test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit
        this feature, learning a deterministic pre-defined sequence of
        actions.
    """
    super().__init__(env)
    assert isinstance(env.observation_space, gym.spaces.Box)

    low, high = env.observation_space.low, env.observation_space.high
    low, high = np.concatenate((low, [0.])), np.concatenate((high, [1.]))
    
    self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    if (isinstance(env, TimeLimit)):
      self._max_steps = env._max_episode_steps
    else:
      self._max_steps = max_steps
    
    self._current_step = 0
    self._test_mode = test_mode

  def reset(self, **kwargs) -> np.ndarray:
    self._current_step = 0
    obs = self.env.reset(**kwargs)
    return self._get_obs(obs)

  def step(self, action: np.ndarray) -> Tuple[Any]:
    self._current_step += 1
    obs, reward, done, info = self.env.step(action)
    return self._get_obs(obs), reward, done, info

  def _get_obs(self, obs: np.ndarray) -> np.ndarray:
    """
    Concatenate the time feature to the current observation.
    :param obs: (np.ndarray)
    :return: (np.ndarray)
    """
    # Remaining time is more general
    time_feature = 1 - (self._current_step / self._max_steps)
    if self._test_mode:
      time_feature = 1.0
    # Optionnaly: concatenate [time_feature, time_feature ** 2]
    return np.concatenate((obs, [time_feature])).astype(obs.dtype)
