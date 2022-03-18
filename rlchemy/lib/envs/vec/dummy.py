# --- built in ---
from typing import (
  Any,
  Callable,
  List
)
# --- 3rd party ---
import gym
# --- my module ---
from rlchemy.lib.envs.vec import base as vec_base

__all__ = [
  'DummyVecEnv',
  'VecEnv'
]

class EnvWorker(vec_base.BaseEnvWorker):
  def __init__(self, env_fn: Callable, auto_reset: bool):
    self.env = env_fn()
    self._res = None
    super().__init__(env_fn, auto_reset)
  
  def getattr(self, attrname: str) -> Any:
    return getattr(self.env, attrname)

  def setattr(self, attrname: str, value: Any) -> Any:
    return setattr(self.env, attrname, value)

  def reset(self, **kwargs) -> Any:
    return self.env.reset(**kwargs)

  def step_async(self, act: Any):
    obs, rew, done, info = self.env.step(act)
    if self._auto_reset and done:
      obs = self.env.reset()
    self._res = (obs, rew, done, info)

  def step_wait(self) -> Any:
    return self._res

  def seed(self, seed: int) -> Any:
    super().seed(seed)
    return self.env.seed(seed)

  def render(self, **kwargs) -> Any:
    return self.env.render(**kwargs)

  def close_async(self):
    self.env.close()

  def close_wait(self):
    pass

class DummyVecEnv(vec_base.BaseVecEnv):
  def __init__(
    self,
    env_fns: List[Callable],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    super().__init__(env_fns, EnvWorker, **kwargs)

class VecEnv(vec_base.BaseVecEnv):
  def __init__(
    self,
    envs: List[gym.Env],
    **kwargs
  ):
    kwargs.pop('worker_class', None)
    env_fns = [lambda i=j: envs[i] for j in range(len(envs))]
    super().__init__(env_fns, EnvWorker, **kwargs)