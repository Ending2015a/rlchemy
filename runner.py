# --- built in ---
from typing import Any, Optional
import collections
# --- 3rd party ---
import numpy as np
# --- my module ---
import rlchemy as rl

class Runner():
  def __init__(
    self,
    env: rl.envs.BaseVecEnv,
    agent,
    buffer,
    last_n_episodes: int = 10
  ):
    assert isinstance(env, rl.envs.BaseVecEnv), \
      "`env` must be a vectorized environment."
    assert last_n_episodes > 0
    self.env = env
    self.agent = agent
    self.buffer = buffer
    self.last_n_episodes = 10

    self.total_timesteps = 0
    self._cached_obs: Any = None
    self._cached_states: Optional[Any] = None
    self._cached_masks: Optional[Any] = None
    self._cached_rollout: Optional[Any] = None
    # statistics
    # TODO future: get statistics from Monitor
    # we also check that if a Monitor is attached
    # to the environments. If not, we disable the
    # runner statistics
    self.last_n_lengths = None
    self.last_n_rewards = None
    self.episode_lengths = None
    self.episode_rewards = None
    self.completed_episodes = 0

  def make_empty_masks(self):
    # all True
    return np.ones((self.env.n_envs,), dtype=bool)

  def reset_statistics(self):
    # TODO future: get those values from Monitor
    self.last_n_lengths = collections.deque(maxlen=self.last_n_episodes)
    self.last_n_rewards = collections.deque(maxlen=self.last_n_episodes)
    self.episode_lengths = np.zeros((self.env.n_envs,), dtype=np.int32)
    self.episode_rewards = np.zeros((self.env.n_envs,), dtype=np.float32)

  def reset(self):
    self._cached_obs: Any = self.env.reset()
    # get agent's internal states
    # the states are not None only if the agent
    # has an internal states, e.g. recurrent nets
    states = self.agent.get_states(batch_size=self.env.n_envs)
    masks = self.make_empty_masks()
    if states is not None:
      self._cached_states = states
      self._cached_masks = masks
    self.reset_statistics()

  def _sample_random_actions_and_states(self):
    assert self._cached_obs is not None, \
      "Runner has not reset yet. Call Runner.reset() first."
    obs = self._cached_obs
    states = self._cached_states
    masks = self._cached_masks
    # random sample actions
    act = np.asarray([s.sample() for s in self.env.action_spaces])
    if states is not None:
      _, next_states = self.agent.predict(
        obs,
        states,
        masks,
        proc_act = False,
        det = False
      )
    else:
      next_states = None
    # TODO: change this to self.agent.rev_proc_act(act)
    rawact = rl.utils.normalize(
      act,
      low = self.env.action_space.low,
      high = self.env.action_space.high,
      nlow = -1.0,
      nhigh = 1.0
    )
    return act, rawact, next_states

  def _sample_predicted_actions_and_states(self):
    assert self._cached_obs is not None, \
      "Runner has not reset yet. Call Runner.reset() first."
    obs = self._cached_obs
    states = self._cached_states
    masks = self._cached_masks
    rawact, next_states = self.agent.predict(
      obs,
      states,
      masks,
      proc_act = False,
      det = False
    )
    # change this to self.agent.proc_act(rawact)
    act = rl.utils.denormalize(
      rawact,
      low = self.action_space.low,
      high = self.action_space.high,
      nlow = -1.0,
      nhigh = 1.0
    )
    return act, rawact, next_states

  def _collect_step(self, random: bool=False):
    # TODO: you can customize this function in your agent.
    obs = self._cached_obs
    states = self._cached_states
    masks = self._cached_masks
    if random:
      act, rawact, next_states = \
        self._sample_random_actions_and_states()
    else:
      act, rawact, next_states = \
        self._sample_random_actions_and_states()
    # step environment
    next_obs, rew, done, info = self.env.step(act)
    one_sample = dict(
      obs = obs,
      act = rawact,
      done = done,
      rew = rew
    )
    if states is not None:
      one_sample['states'] = states
      one_sample['masks'] = masks
    # add samples to the buffer
    self.buffer.add(**one_sample)
    # make next states
    # the next states should be reset if done=True
    next_states = self.agent.get_states(
      batch_size = self.env.n_envs,
      states = next_states,
      masks = done
    )
    # cache rollouts
    self._cached_obs = next_obs
    self._cached_states = next_states
    self._cached_masks = done
    return next_obs, rew, done, info

  def step(self, random: bool=False):
    if hasattr(self.agent, "runner_collect_step") and \
        callable(self.agent.runner_collect_step):
      next_obs, rew, done, info = self.agent.runner_collect_step(self)
    else:
      next_obs, rew, done, info = self._collect_step(random=random)
    # make statistics
    self.total_timesteps += self.env.n_envs
    assert len(rew) == len(self.episode_rewards)
    self.episode_lengths += 1
    self.episode_rewards += rew
    for env_idx, episode_done in enumerate(done):
      if episode_done:
        self.last_n_rewards.append(self.episode_rewards[env_idx])
        self.last_n_lengths.append(self.episode_lengths[env_idx])
        self.completed_episodes += 1
    # reset statistics which is done
    self.episode_lengths *= 1 - done
    self.episode_rewards *= 1 - done
    self._cached_rollout = (next_obs, rew, done, info)
    return next_obs, rew, done, info

  def get_log_dict(self):
    # TODO future: get those values from Monitor
    # also we disable this function if Monitor is not
    # attached to the env
    return {
      'last_episode_rewards': self.last_n_rewards[-1],
      'last_episode_length': self.last_n_lengths[-1],
      'avg_episode_rewards': np.mean(self.last_n_rewards),
      'avg_episode_length': np.mean(self.last_n_lengths),
      'completed_episodes': self.completed_episodes
    }

