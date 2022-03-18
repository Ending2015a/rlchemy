# --- built in ---
from dataclasses import dataclass
from typing import (
  Any,
  Dict,
  List,
  Optional,
)
import copy
# --- 3rd party ---
import gym
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import omegaconf
from omegaconf import OmegaConf
# --- my module ---
import rlchemy as rl
from runner import Runner

class DiagGaussianPolicyNet(rl.nets.DiagGaussianPolicyNet):
  def create_logstd_model(self, dim, out_dim):
    """Clamped logstd model"""
    return nn.Sequential(
        nn.Linear(dim, out_dim),
        rl.nets.Lambda(lambda x: torch.clamp(x, -20, 2))
    )

class PolicyNet(rl.nets.PolicyNet):
  support_spaces = [gym.spaces.Box]
  def create_gaussian_policy(self, dim):
    model = DiagGaussianPolicyNet(dim, self.action_space, self.squash)
    return model

class ValueNets(rl.nets.MultiHeadValueNets):
    pass

class Agent(nn.Module):
  def __init__(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    squash: bool = True,
    n_critics: int = 2,
    mlp_units: list = [256, 256],
    force_mlp: bool = False,
    activ: str = 'relu'
  ):
    super().__init__()
    self.observation_space = observation_space
    self.action_space = action_space
    self.squash = squash
    self.n_critics = n_critics
    self.mlp_units = mlp_units
    self.force_mlp = force_mlp
    self.activ = activ

    self.policy = None
    self.values = None
    self._obs_space_low_th = None
    self._act_space_low_th = None
    self._act_space_low_th = None
    self._act_space_high_th = None

  @property
  def device(self):
    p = next(iter(self.parameters()), None)
    return p.device if p is not None else torch.device('cpu')

  def setup_spaces(self):
    # TODO future: safety check
    self._obs_space_low_th = nn.Parameter(
      rl.utils.to_tensor(self.observation_space.low),
      requires_grad = False
    )
    self._obs_space_high_th = nn.Parameter(
      rl.utils.to_tensor(self.observation_space.high),
      requires_grad = False
    )
    self._act_space_low_th = nn.Parameter(
      rl.utils.to_tensor(self.action_space.low),
      requires_grad = False
    )
    self._act_space_high_th = nn.Parameter(
      rl.utils.to_tensor(self.action_space.high),
      requires_grad = False
    )
    self._act_bounded = rl.utils.is_bounded(self.action_space)

  def setup_model(self):
    # make feature extractors
    make_net_fn = lambda space: rl.nets.AwesomeNet(
      input_space = space,
      force_mlp = self.force_mlp,
      mlp_units = self.mlp_units,
      activ = self.activ
    )
    base_net_pi = make_net_fn(self.observation_space)
    base_net_vfs = rl.nets.Parallel([
      make_net_fn((self.observation_space, self.action_space))
      for n in range(self.n_critics)
    ])
    # create policy and value nets
    self.policy = nn.Sequential(
      base_net_pi,
      PolicyNet(
        action_space = self.action_space,
        squash = self.squash
      )
    )
    self.values = nn.Sequential(
      base_net_vfs,
      ValueNets(n_heads=self.n_critics)
    )
    # forward modules to create networks
    obs_inputs = rl.utils.input_tensor(self.observation_space, dtype=torch.float32)
    act_inputs = rl.utils.input_tensor(self.action_space, dtype=torch.float32)
    self.forward_policy(obs_inputs, proc_obs=True)
    self.forward_value((obs_inputs, act_inputs), proc_obs=True)

  def setup(self):
    self.setup_spaces()
    self.setup_model()

  def proc_observation(self, obs):
    return obs.to(dtype=torch.float32)

  def proc_action(self, act):
    """Post process actions"""
    if isinstance(self.action_space, gym.spaces.Box):
      if self.squash:
        act = torch.clamp(act, -1., 1.)
        if self._act_bounded:
          low = self._act_space_low_th
          high = self._act_space_high_th
          act = rl.utils.normalize(act, -1., 1., low, high)
      else:
        act = torch.clamp(act, low, high)
    return act

  def forward_policy(self, x, proc_obs=True, **kwargs):
    #TODO: support rnn
    if proc_obs:
      x = self.proc_observation(x)
    return self.policy(x)

  def forward_value(self, x, proc_obs=True, **kwargs):
    #TODO: support rnn
    obs, act = x
    if proc_obs:
      obs = self.proc_observation(obs)
    outputs = self.values((obs, act)) # (b, 1) * n
    return outputs

  def forward(
    self,
    x: torch.Tensor,
    states = None,
    masks = None,
    proc_obs: bool = True,
    proc_act: bool = False,
    det: bool = False
  ) -> torch.Tensor:
    #TODO: support rnn
    dist = self.forward_policy(x, proc_obs=proc_obs)
    # predict actions
    if det:
      act = dist.root.mode()
    else:
      act = dist.root.sample()
    act, logp = dist.inject_log_prob(act)
    if proc_act:
      act = self.proc_action(act)
    return act, logp

  @torch.no_grad()
  def _predict(
    self,
    x: np.ndarray,
    *args,
    **kwargs
  ) -> np.ndarray:
    # we dont need to modify rnn states here
    is_batch = rl.utils.is_batch_sample(x, self.observation_space)
    if not is_batch:
      expand_op = lambda x: np.expand_dims(x, axis=0)
      x = rl.utils.map_nested(x, expand_op)
    # predict actions
    x = rl.utils.nested_to_tensor(x, device=self.device)
    outputs, *_ = self(x, *args, **kwargs)
    outputs = rl.utils.nested_to_numpy(outputs)
    if not is_batch:
      squeeze_op = lambda x: np.squeeze(x, axis=0)
      outputs = rl.utils.map_nested(outputs, squeeze_op)
    return outputs

  def predict(
    self,
    x: np.ndarray,
    states = None,
    masks = None,
    proc_obs: bool = True,
    proc_act: bool = True,
    det: bool = True
  ) -> np.ndarray:
    #TODO: support rnn
    return self._predict(
      x,
      states = states,
      masks = masks,
      proc_obs = proc_obs,
      proc_act = proc_act,
      det = det
    )

  def update(self, other_model, tau=1.0):
    tar = list(self.parameters())
    src = list(other_model.parameters())
    rl.utils.soft_update(tar, src, tau=tau)


class SAC(pl.LightningModule):
  def __init__(
    self,
    config: dict,
    env: Optional[rl.envs.BaseVecEnv] = None,
    observation_space: Optional[gym.spaces.Space] = None,
    action_space: Optional[gym.spaces.Space] = None
  ):
    super().__init__()
    # initialize
    self.env = None
    self.observation_space = None
    self.action_space = None
    self.buffer = None
    self.sampler = None
    self.agent = None
    self.agent_tar = None
    self.log_alpha = None
    self.target_ent = None
    self._train_runner = None
    self._optim_mapping = {}
    self._model_setup = False
    self._train_batch_loader = None
    self._cache_rollout = None

    # initialize lightning module
    self.config = OmegaConf.create(config)
    self.set_envs(env=env)
    observation_space = self.observation_space
    action_space = self.action_space
    self.save_hyperparameters(ignore=['env'])
    self.automatic_optimization = False

    self.maybe_setup_model()

  def set_envs(
    self,
    env: Optional[rl.envs.BaseVecEnv] = None
  ):
    if env is not None:
      self.env = env
      self.set_spaces(env.observation_space, env.action_space)
  
  def set_spaces(
    self,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space
  ):
    # check observation/action spaces
    self.observation_space = observation_space
    self.action_space = action_space

  def maybe_setup_model(self):
    if (not self._model_setup
        and self.observation_space is not None
        and self.action_space is not None):
      self.setup(stage='model')

  def setup_model(self):
    self.buffer = rl.data.ReplayBuffer(
      **self.config.buffer
    )
    self.sampler = rl.data.UniformSampler(self.buffer)
    self.agent = Agent(
      self.observation_space,
      self.action_space,
      **self.config.agent
    )
    self.agent.setup()
    self.agent_tar = Agent(
      self.observation_space,
      self.action_space,
      **self.config.agent
    )
    self.agent_tar.setup()
    self.agent_tar.update(self.agent, tau=1.0)
    # setup entropy coefficient
    self.log_alpha = nn.Parameter(
      torch.tensor(np.log(self.config.init_alpha))
    )
    if self.target_ent is None:
      self.target_ent = -np.prod(self.action_space.shape)
    self.target_ent = float(np.asarray(self.target_ent).item())

  def train_batch_fn(self):
    # sample n steps for every epoch
    for _ in range(self.config.n_steps):
      self._train_runner.step(random=False)
    # generate n batches for every epoch
    for _ in range(self.config.n_gradsteps):
      #TODO: sample sequence
      batch = self.sampler(self.config.batch_size)
      batch['next_obs'] = self.sampler.rel[1]['obs']
      if 'states' in batch:
        batch['next_states'] = self.sampler.rel[1]['states']
        batch['next_masks'] = self.sampler.rel[1]['masks']
      yield batch

  def setup_train(self):
    assert self.env is not None
    self._train_runner = Runner(
      env = self.env,
      agent = self,
      buffer = self.buffer,
      last_n_episodes = self.config.last_n_episodes
    )
    self._train_runner.reset()
    for i in range(self.config.warm_steps):
      self._train_runner.step(random=True)

  def get_states(self, states=None, masks=None, batch_size=None):
    return None

  # === lightning hooks ===

  def setup(self, stage: str):
    # setup env, model, config here
    # stage: either 'fit', 'validate'
    # 'test' or 'predict'

    if stage == 'model':
      self.setup_model()
      self._model_setup = True
    
    if stage == 'fit':
      self.setup_train()

  def configure_optimizers(self):
    optim_dict = self.create_optimizers()
    optims = []
    for index, (name, optim) in enumerate(optim_dict.items()):
      self._optim_mapping[name] = index
      optims.append(optim)
    return tuple(optims)

  def forward(
    self,
    x: torch.Tensor,
    states: Optional[Any] = None,
    masks: Optional[Any] = None,
    proc_obs: bool = True,
    proc_act: bool = False,
    det: bool = False
  ) -> torch.Tensor:
    # predict
    return self.agent(
      x,
      states = states,
      masks = masks,
      proc_obs = proc_obs,
      proc_act = proc_act,
      det = det
    )

  def predict(
    self,
    x: Any,
    states: Optional[Any] = None,
    masks: Optional[Any] = None,
    proc_obs: bool = True,
    proc_act: bool = True,
    det: bool = True
  ):
    return self.agent.predict(
      x,
      states = states,
      masks = masks,
      proc_obs = proc_obs,
      proc_act = proc_act,
      det = det
    )

  def training_step(self, batch, batch_idx):
    log_dict = self._train_value(batch, batch_idx)
    log_dict.update(self._train_policy(batch, batch_idx))
    self.agent_tar.update(self.agent, tau=self.config.tau)
    self.log_dict(log_dict, sync_dist=True)
    self._train_runner.log_dict(self, scope='train_runner')
  
  def train_dataloader(self):
    dataset = rl.data.BatchProvider(self.train_batch_fn)
    # we set batch_size=None here to enable manual batching
    # see: https://pytorch.org/docs/stable/data.html#disable-automatic-batching
    return DataLoader(dataset=dataset, batch_size=None)

  # === not lightning hooks ===

  def create_optimizers(self):
    return dict(
      policy_optim = torch.optim.Adam(
        self.agent.policy.parameters(),
        lr = self.config.policy_lr
      ),
      value_optim = torch.optim.Adam(
        self.agent.values.parameters(),
        lr = self.config.value_lr
      ),
      alpha_optim = torch.optim.Adam(
        [self.log_alpha],
        lr = self.config.alpha_lr
      )
    )

  def get_optimizer(self, name):
    optimizers = self.optimizers()
    index = self._optim_mapping[name]
    return optimizers[index]

  def alpha_loss(self, logp):
    tar = logp + self.target_ent
    return -1.0 * torch.mean(self.log_alpha.exp() * tar.detach())

  def policy_loss(self, logp, qs):
    alpha = self.log_alpha.exp()
    target_q = torch.min(qs, dim=0)[0] # (b, 1)
    return torch.mean(alpha * logp - target_q)
  
  def value_loss(
    self,
    obs: torch.Tensor,
    act: torch.Tensor,
    done: torch.Tensor,
    rew: torch.Tensor,
    next_obs: torch.Tensor,
    states: Optional[Any] = None,
    masks: Optional[Any] = None,
    next_states: Optional[Any] = None,
    next_masks: Optional[Any] = None
  ) -> Dict[str, torch.Tensor]:
    act = act.to(dtype=torch.float32)
    rew = rew.to(dtype=torch.float32)
    done = done.to(dtype=torch.float32)
    rew = torch.unsqueeze(rew, dim=-1)
    done = torch.unsqueeze(done, dim=-1)
    with torch.no_grad():
      alpha = self.log_alpha.exp()
      next_act, next_logp = self.agent(
        next_obs,
        states = states,
        masks = masks,
        proc_obs = True,
        proc_act = False,
        det = False
      ) # (b, )
      next_logp = torch.unsqueeze(next_logp, dim=-1) # (b, 1)
      # calculate target q values
      next_qs = self.agent_tar.forward_value(
        (next_obs, next_act),
        states = next_states,
        masks = next_masks
      ) # (n, b, 1)
      next_qs = torch.stack(next_qs, dim=0) # (n, b, 1)
      next_q = torch.min(next_qs, dim=0)[0] # (b, 1)
      next_v = next_q - alpha * next_logp
      y = rew + (1.-done) * self.config.gamma * next_v # (b, 1)
    # calculate current q
    qs = self.agent.forward_value(
      (obs, act),
      states = states,
      masks = masks
    ) # n * (b, 1)
    qs = torch.stack(qs, dim=0)
    tds = y - qs # (n, b, 1)
    losses = rl.loss.regression(tds, loss_type=self.config.loss_type)
    loss = torch.mean(losses.sum(dim=0))
    return loss

  def _train_policy(self, batch, batch_idx):
    # predict actions and log probs
    act, logp = self.agent(
      batch['obs'],
      states = batch.get('states', None),
      masks = batch.get('masks', None),
      proc_obs = True,
      proc_act = False,
      det = False
    )
    logp = torch.unsqueeze(logp, dim=-1) # (b, 1)
    qs = self.agent.forward_value(
      (batch['obs'], act),
      states = batch.get('states', None),
      masks = batch.get('masks', None)
    )
    qs = torch.stack(qs, dim=0)
    # compute policy loss, update policy networks
    policy_optim = self.get_optimizer('policy_optim')
    policy_loss = self.policy_loss(logp, qs)
    policy_optim.zero_grad()
    self.manual_backward(policy_loss)
    policy_optim.step()
    # compute alpha loss, update alpha
    alpha_optim = self.get_optimizer('alpha_optim')
    alpha_loss = self.alpha_loss(logp)
    alpha_optim.zero_grad()
    self.manual_backward(alpha_loss)
    alpha_optim.step()
    return {
      'policy_loss': policy_loss,
      'alpha_loss': alpha_loss,
      'alpha': self.log_alpha.exp(),
      'logp': logp.mean()
    }

  def _train_value(self, batch, batch_idx):
    value_optim = self.get_optimizer('value_optim')
    # compute loss
    loss = self.value_loss(
      obs = batch['obs'],
      act = batch['act'],
      done = batch['done'],
      rew = batch['rew'],
      next_obs = batch['next_obs'],
      states = batch.get('states', None),
      masks = batch.get('masks', None),
      next_states = batch.get('next_states', None),
      next_masks = batch.get('next_masks', None)
    )
    # back propogate
    value_optim.zero_grad()
    self.manual_backward(loss)
    value_optim.step()
    return {
      'value_loss': loss
    }




def main():
  import argparse
  parser = argparse.ArgumentParser(add_help=True)
  parser.add_argument('--config', type=str, default=None, help='configuration file location')
  parser.add_argument('dot_config', metavar='CCC', type=str,
    nargs='*', help='dot-list configurations')
  
  a = parser.parse_args()

  # create configurations
  base_conf = OmegaConf.load(a.config)
  dot_conf = OmegaConf.from_dotlist(a.dot_config)
  conf = OmegaConf.merge(base_conf, dot_conf)

  env = rl.envs.VecEnv([
    gym.make(conf.env.env_id)
    for i in range(conf.env.n_envs)
  ])
  env.seed(conf.env.seed)

  model = SAC(conf.SAC, env=env)
  # equivalent
  # model.set_envs(env=env)

  checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor = "train_runner/avg_episode_rewards",
    save_last = True,
    save_top_k = 5,
    mode = "max",
    every_n_epochs = 10000,
    verbose = True
  )
  pl.seed_everything(10)
  trainer = pl.Trainer(
    **conf.trainer,
    deterministic = True,
    callbacks = checkpoint_callback
  )
  trainer.fit(model)

if __name__ == '__main__':
  main()