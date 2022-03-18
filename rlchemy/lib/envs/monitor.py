# --- bulid in ---
import gc
import os
import abc
import csv
import copy
import json
import time
import logging
import inspect
import tempfile
import distutils
import subprocess
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Optional,
  Tuple,
  Union
)
# --- 3rd party ---
import gym
import numpy as np
import pandas as pd
# --- my module ---
import rlchemy
from rlchemy.lib import utils as rl_utils

__all__ = [
  'load_monitor',
  'load_monitors',
  'load_trajectory',
  'Monitor',
  'MonitorToolChain',
  'StatsRecorder',
  'VideoRecorder',
  'TrajectoryRecorder'
]

def log_message(message: Any, level='WARNING'):
  prevframe = inspect.currentframe().f_back
  frameinfo = inspect.getframeinfo(prevframe)
  modulepath = os.path.dirname(rlchemy.__file__)
  filename = os.path.relpath(frameinfo.filename, modulepath)
  level = logging.getLevelName(level)
  logging.log(level, f"{level}:{filename}:{frameinfo.lineno}: {message}")

def load_monitor(path: str) -> Tuple[str, pd.DataFrame]:
  """Load monitor file

  Args:
    path (str): the exact file path to a monitor file
  
  Returns:
    dict: header of the monitor file
    pd.DataFrame: loaded data frame
  """
  with open(path, 'rt') as f:
    first_line = f.readline()
    assert first_line[0] == '#'
    header = json.loads(first_line[1:])
    df = pd.read_csv(f, index_col=None)
    df.sort_values("walltime", inplace=True)
    df.reset_index(inplace=True)
  return header, df

def load_monitors(
  paths: List[str],
  align_walltime: bool=True
) -> Tuple[List[str], List[pd.DataFrame]]:
  """Load monitor files and align the starting walltimes
  of all monitors.

  Args:
    paths (list): list of monitor files
    align_walltime (bool, optional): whether to align the wall times.
      Defaults to True.

  Returns:
    list[dict]: list of headers
    list[pd.DataFrame]: list of data frames
  """
  if not paths:
    raise RuntimeError(f"No monitor files provided")
  dfs = []
  headers = []
  for path in paths:
    header, df = load_monitor(path)
    headers.append(header)
    dfs.append(df)
  if align_walltime:
    start = min(header['t_start'] for header in headers)
    for header, df in zip(headers, dfs):
      df['walltime'] += header['t_start']
      df['walltime'] -= start
      header["t_start"] = start
  return headers, dfs

def save_trajectory(path: str, trajectory_data: Any):
  """Save trajectory data to `path`

  Args:
    path (str): save path
    trajectory_data (Any): trajectory data, a nested data type.
  """
  # flatten nested data to list of numpy arrays
  # and save them to a npz file. The nested structure
  # is saved to '_struct'.
  struct, flat_data = rl_utils.unpack_structure(trajectory_data, sortkey=False)
  try:
    # save to file
    np.savez(path, *flat_data, _struct=struct)
  except FileNotFoundError:
    logging.error(f"Failed to write trajectory data to path: {path}\n"
      "This error may caused by trying to write files in a __del__ "
      "function.")

def load_trajectory(path: str, allow_pickle: bool=True) -> Any:
  """Load trajectory data from `path`

  Args:
    path (str): saved path

  Returns:
    dict: trajectory data, keys:
      obs: observations,
      act: actions,
      next_obs: next observations,
      rew: rewards,
      done: done flags,
      info: extra informations.
  """
  npdata = np.load(path, allow_pickle=allow_pickle)
  if '_struct' not in npdata.files:
    raise ValueError(
      f"Trajectory data does not contain '_struct': {path}"
    )
  struct = npdata['_struct'].item()
  npdata.files.remove('_struct')
  flat_data = [npdata[arr] for arr in npdata.files]
  # recover trajectory data from list of numpy arrays
  data = rl_utils.pack_sequence(struct, flat_data)
  return data

# === Monitor ===

class Monitor(gym.Wrapper):
  def __init__(
    self,
    env: gym.Env,
    root_dir: str = './monitor',
    prefix: Optional[str] = None,
    allow_early_reset: bool = True,
    video: bool = False,
    video_kwargs: dict = {}
  ):
    """A monitor wrapper to record environment episode reward, length, time
    and game screen.

    Args:
      env (gym.Env): Environment.
      root_dir (str, optional): Root path to store monitor files. Defaults 
        to './monitor'.
      prefix (str, optional): Monitor file prefix. Defaults to None.
      allow_early_reset (bool, optional): If False, monitor ignores the
        episodes that are reset before done. Defaults to True.
      video (bool, optional): Whether to record video (mp4). This feature 
        only works for the environments which support 'rgb_array' render 
        mode. Defaults to False.
      video_kwargs (dict, optional): Recording keywarded arguments. See
        VideoRecorder for more info. Defaults to {}.
    """
    super().__init__(env)

    self.root_dir = root_dir or './'
    self.prefix   = prefix
    self.env_id   = env.unwrapped.spec.id
    self.allow_early_reset = allow_early_reset
    # storing monitor tools
    self._tools = []
    # In default, the StatsRecorder is enabled
    # and it must be the first monitor tool to
    # be executed.
    self._stats_recorder = StatsRecorder(
      root_dir          = self.root_dir,
      prefix            = self.prefix,
      ext               = None,
      allow_early_reset = self.allow_early_reset
    )
    self.add_tool(self._stats_recorder)
    # Setup video recorder
    if video:
      video_kwargs = video_kwargs or {}
      self.add_tool(VideoRecorder(**video_kwargs))
  
  @property
  def stats(self) -> rl_utils.StateObject:
    """Return StateRecorder"""
    return self._stats_recorder._stats

  @property
  def tools(self) -> List['MonitorToolChain']:
    """Return Toolchain"""
    return self._tools

  def add_tool(self, tool: 'MonitorToolChain') -> 'Monitor':
    """Add monitor tool"""
    if hasattr(tool, 'set_monitor'):
      tool.set_monitor(self)
    self._tools.append(tool)
    return self

  def add_tools(self, tool_list: list) -> 'Monitor':
    """Add monitor toolchain"""
    for tool in tool_list:
      self.add_tool(tool)
    return self

  def step(self, act: Any) -> Any:
    act = self._before_step(act)
    obs, rew, done, info = self.env.step(act)
    return self._after_step(obs, rew, done, info)

  def reset(self, **kwargs) -> Any:
    kwargs = self._before_reset(**kwargs)
    obs = self.env.reset(**kwargs)
    return self._after_reset(obs)

  def close(self):
    for tool in self._tools:
      tool.close()
    super().close()
  
  def _before_step(self, act: Any) -> Any:
    for tool in self._tools:
      act = tool._before_step(act)
    return act
  
  def _after_step(self, *data) -> Any:
    for tool in self._tools:
      data = tool._after_step(*data)
    return data

  def _before_reset(self, **kwargs) -> Any:
    for tool in self._tools:
      kwargs = tool._before_reset(**kwargs)
    return kwargs

  def _after_reset(self, obs) -> Any:
    for tool in self._tools:
      obs = tool._after_reset(obs)
    return obs

class MonitorToolChain(metaclass=abc.ABCMeta):
  def __init__(self):
    """A base class for implementing custom monitor tool chains"""
    self._monitor = None
    self._stats   = None
    self._env     = None

  def set_monitor(self, monitor: Monitor):
    """This function is called when this object is added
    to some Monitor object.

    Args:
      monitor (Monitor): the monitor this object added to.
    """
    self._monitor = monitor
    self._stats   = monitor.stats
    self._env     = monitor.env

  @property
  def stats(self) -> rl_utils.StateObject:
    if self._monitor is None:
      raise RuntimeError("This monitor tool is not attached to "
        "a monitor.")
    return self._monitor.stats

  def _before_step(self, act: Any) -> Any:
    return act

  def _after_step(self, *data) -> Any:
    return data

  def _before_reset(self, **kwargs) -> Any:
    return kwargs

  def _after_reset(self, obs) -> Any:
    """The episode info (stats) updates here"""
    return obs

  def close(self):
    pass

# === Stats Recorder ===

class Stats(rl_utils.StateObject):
  def __init__(self):
    super().__init__()
    self.episodes    = 0  # Cirrent episode number
    self.steps       = 0  # Current step number
    self.start_steps = 0  # Step when this episode begins
    self.rewards     = 0  # Cumulative rewards in one episode
    self.ep_rewards  = [] # Rewards per episode
    self.ep_lengths  = [] # Episode lengths per episode
    self.ep_times    = [] # Episode time per episode

class StatsRecorder(MonitorToolChain):
  monitor_ext = 'monitor.csv'
  def __init__(
    self,
    root_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    ext: Optional[str] = None,
    allow_early_reset: bool = True
  ):
    """StatsRecorder records environment stats: episodic rewards,
    total timesteps, time spent. And write them to a monitor file
    `{root_dir}/{prefix}.{ext}` in CSV format.

    CSV contents:
    The first line is a JSON comment indicating the timestamp 
    the environment started, and the name of the environment.
    The next line is the header line, indicating the column
    names [r, l, t] for each columns, where 'r' denotes
    rewards, 'l' episode length, 't' episode running time.
    The following lines are data for each episode.

    Example:
      # {t_start: timestamp, env_id: env_id}
      r, l, t
      320, 1000, 5.199693
      ...

    r: total episodic rewards
    l: episode length
    t: running time (time - t_start)

    Args:
      root_dir (str, optional): Root path to save stats files. Defaults 
        to None.
      prefix (str, optional): File prefix. Defaults to None.
      ext (str, optional): File extention. Defaults to 'monitor.csv'.
      env_id (str, optional): Environment ID. Defaults to None.
    """
    self.t_start = time.time()

    self.root_dir = root_dir
    self.prefix   = prefix
    self.ext      = ext
    self.env_id   = None
    self.allow_early_reset = allow_early_reset

    # initialize
    self._closed      = False
    self._need_reset  = True
    self.filepath     = None
    self.header       = None
    self.file_handler = None
    self.writer       = None

    self._stats = Stats()

  def set_monitor(self, monitor: Monitor):
    self._monitor = monitor
    self.root_dir = self.root_dir or monitor.root_dor
    self.prefix   = self.prefix or monitor.prefix
    self.ext      = self.ext or self.monitor_ext
    self.env_id   = self.env_id or monitor.env_id
    self._setup_writer()

  def _setup_writer(self):
    # setup csv file
    self.filepath = self._make_save_path(self.root_dir, self.prefix, self.ext)
    self.header = json.dumps({'t_start': self.t_start, 'env_id': self.env_id})

    filedir = os.path.dirname(self.filepath)
    relpath = os.path.relpath(self.filepath)

    self._create_path(filedir)

    # write header
    self.file_handler = open(self.filepath, 'wt')
    self.file_handler.write(f'#{self.header}\n')

    # create csv writer
    self.writer = csv.DictWriter(self.file_handler,
      fieldnames=('rewards', 'length', 'walltime'))
    self.writer.writeheader()
    self.file_handler.flush()

  @property
  def stats(self) -> rl_utils.StateObject:
    return self._stats

  @property
  def closed(self) -> bool:
    return self._closed

  def _make_save_path(
    self,
    base_path: str,
    prefix: Optional[str],
    ext: str
  ) -> str:
    """Make full path
    {base_path}/{prefix}.{ext} or
    {base_path}/{ext} if {prefix} is None
    """
    paths = []
    if prefix:
      paths.append(str(prefix))
    paths.append(str(ext))

    filename = '.'.join(paths)
    path = os.path.join(base_path, filename)

    return os.path.abspath(path)

  def _create_path(
    self,
    dirpath: Optional[str]=None,
    filepath: Optional[str]=None
  ):
    """Safely create directories
    
    Args:
      dirpath (str, optional): directory path. Defaults to None.
      filepath (str, optional): file path. Defaults to None.
    """
    rl_utils.safe_makedirs(dirpath=dirpath, filepath=filepath)

  def _before_step(self, act: Any) -> Any:
    if self._need_reset:
      raise RuntimeError('Tried to step environment that needs reset')
    return act

  def _after_step(self, obs: Any, rew: Any, done: Any, info: Any) -> Tuple[Any]:
    # append rewards
    self.stats.rewards += rew
    self.stats.steps += 1
    # episode ended
    if done:
      self._need_reset = True
      ep_info = self._write_results()
      info['monitor_episode'] = ep_info
    return obs, rew, done, info

  def _before_reset(self, **kwargs) -> Dict:
    """Do nothing"""
    return kwargs

  def _after_reset(self, obs: Any) -> Any:
    """Reset episodic info"""
    if not self._need_reset:
      if self.allow_early_reset:
        self._write_results()
    self.stats.rewards = 0
    self.stats.start_steps = self.stats.steps
    self.stats.episodes += 1
    self._need_reset = False
    return obs

  def _write_results(self) -> Dict:
    self._need_reset = True
    ep_rew = self.stats.rewards
    ep_len = self.stats.steps - self.stats.start_steps
    ep_time = time.time()-self.t_start
    ep_info = {
      'rewards': round(ep_rew, 6),
      'length': ep_len,
      'walltime': round(ep_time, 6)
    }
    self.stats.ep_rewards.append(ep_rew)
    self.stats.ep_lengths.append(ep_len)
    self.stats.ep_times.append(ep_time)
    # write episode info to csv
    if self.writer:
      self.writer.writerow(ep_info)
      self.file_handler.flush()
    return ep_info

  def close(self):
    if self.file_handler is not None:
      self.file_handler.close()
    self._closed = True

  def __del__(self):
    if not self._closed:
      log_message(
        "The StatsRecorder is not closed, please "
        "ensure you have closed the environment before the "
        "program ended `env.close()`",
        level='WARNING'
      )
      self.close()

  def flush(self):
    pass

# === Video Recorder ===

class _VideoEncoder(object):
  def __init__(
    self,
    path: str,
    frame_shape: Tuple[int, int, int],
    video_shape: Tuple[int, int, int],
    in_fps: int,
    out_fps: int
  ):
    """Create ffmpeg video encoder

    Args:
      path (str): output path
      frame_shape (Tuple[int,int,int]): input frame shape (h,w,c)
      video_shape (Tuple[int,int,int]): output video shape (h,w,c)
      in_fps (int): input framerate
      out_fps (int): output video framerate
    """
    self.path = path
    self.frame_shape = frame_shape
    self.video_shape = video_shape
    self.in_fps = in_fps
    self.out_fps = out_fps

    self.proc = None
    h,w,c = self.frame_shape
    oh,ow,_ = self.video_shape
    
    if c != 3 and c != 4:
      raise RuntimeError(f'Your frame has shape {self.frame_shape}, but we require (h,w,3) or (h,w,4)')
    
    self.wh = (w,h)
    self.output_wh = (ow,oh)
    self.includes_alpha = (c == 4)

    if distutils.spawn.find_executable('ffmpeg') is not None:
      self.backend = 'ffmpeg'
    else:
      raise RuntimeError('ffmpeg not found')
    
    self._start_encoder()

  def _start_encoder(self):
    self.cmdline = (
      self.backend,
      '-nostats',   # disable encoding progress info
      '-loglevel', 'error', # suppress warnings
      '-y',  # replace existing file
      # input stream settings
      '-f', 'rawvideo', # input format
      '-s:v', "{:d}x{:d}".format(*self.wh), # input shape (w,h)
      '-pix_fmt', ('rgb32' if self.includes_alpha else 'rgb24'),
      '-framerate', '{:d}'.format(self.in_fps), # input framerate,
      '-i', '-', # input from stdin
      # output stream settings
      '-vf', 'scale={:d}:{:d}'.format(*self.output_wh),
      '-c:v', 'libx264', # video codec
      '-pix_fmt', 'yuv420p',
      '-r', '{:d}'.format(self.out_fps),
      self.path
    )
    cmd = ' '.join(self.cmdline)
    #LOG.debug(f'Starting ffmpeg with cmd "{cmd}"')
    self.proc = subprocess.Popen(self.cmdline, stdin=subprocess.PIPE)
  
  @property
  def version_info(self) -> Dict:
    return {
      'backend': self.backend,
      'version': str(subprocess.check_output([self.backend, '-version'],stderr=subprocess.STDOUT)),
      'cmdline': self.cmdline
    }

  @property
  def video_info(self) -> Dict:
    return {
      'width': self.wh[0],
      'height': self.wh[1],
      'input_fps': self.in_fps,
      'output_fps': self.out_fps
    }

  def capture_frame(self, frame: np.ndarray):
    if not isinstance(frame, (np.ndarray, np.generic)):
      raise RuntimeError(f"Wrong type {type(frame)} for {frame} "
        "(must be np.ndarray or np.generic)")
    if frame.shape != self.frame_shape:
      raise RuntimeError(f"Your frame has shape {frame.shape}, "
        "but the VideoRecorder is configured for shape "
        f"{self.frame_shape}")
    if frame.dtype != np.uint8:
      raise RuntimeError(f"Your frame has data type {frame.dtype}, "
        "but we require unit8 (i.e. RGB values from 0-255)")

    self.proc.stdin.write(frame.tobytes())

  def __del__(self):
    if self.proc is not None:
      log_message(
        "WARNING: The _VideoEncoder is not closed, please "
        "ensure you have closed the environment before the "
        "program ended `env.close()`",
        level="WARNING"
      )
      self.close()

  def close(self):
    if self.proc is None:
      return
    
    self.proc.stdin.close()
    ret = self.proc.wait()
    if ret != 0:
      logging.error(f"VideoRecorder encoder exited with status {ret}")

    self.proc = None


class _VideoRecorder(object):
  def __init__(
    self,
    env: gym.Env,
    path: str,
    meta_path: str,
    metadata: Optional[dict] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    in_fps: Optional[int] = None,
    out_fps: Optional[int] = None,
    enabled: Optional[bool] = True
  ):
    """VideoRecorder captures video frames from `env` and encodes it as a
    video with _VideoEncoder and saves to `path`.

    Args:
      env (gym.Emv): Environment.
      path (str): Video save path.
      meta_path (str): Metadata save path.
      metadata (dict, optional): Additional metadata. Defaults to None.
      width (int, optional): Video width. Defaults to None.
      height (int, optional): Video height. Defaults to None.
      in_fps (int, optional): Environment fps. Defaults to None.
      out_fps (int, optional): Output video fps. Defaults to None.
      enabled (bool, optional): Enable this recorder. Defaults to True.
    """
    self.env = env
    self.path = path
    self.meta_path = meta_path
    self.width = width
    self.height = height
    self.in_fps = in_fps
    self.out_fps = out_fps
    self.enabled = enabled
    self.metadata = metadata or {}
    if not self.enabled:
      return

    if not path:
      raise ValueError(f"'path' not specified: {path}")
    if not meta_path:
      raise ValueError(f"'meta_path' not specified: {meta_path}")

    in_fps = env.metadata.get('video.frames_per_second', 30)
    out_fps = env.metadata.get('video.output_frames_per_second', in_fps)
    self.in_fps = self.in_fps or self.out_fps or in_fps
    self.out_fps = self.out_fps or self.in_fps or out_fps
    self._async = env.metadata.get('semantics.async')
    modes = env.metadata.get('render.modes', [])

    if 'rgb_array' not in modes:
      logging.warning(f"Disabling video recorder because {env} not "
          "supports render mode 'rgb_array'")
      self.enabled = False
      return

    # render settings: width, height, ...etc
    self._render_kwargs = {}
    # set frame width, height of the environment
    if width is not None:
      # pybullet_envs:
      if hasattr(env.unwrapped, '_render_width'):
        env.unwrapped._render_width = width
      # mujoco:
      if 'mujoco' in env.unwrapped.spec.entry_point:
        self._render_kwargs['width'] = width
    if height is not None:
      # pybullet_envs:
      if hasattr(env.unwrapped, '_render_height'):
        env.unwrapped._render_height = height
      # mujoco:
      if 'mujoco' in env.unwrapped.spec.entry_point:
        self._render_kwargs['height'] = height
    # Create directory
    self._create_path(filepath=os.path.abspath(self.path))
    self._create_path(filepath=os.path.abspath(self.meta_path))
    
    # Write metadata
    self.metadata['content_type'] = 'video/mp4'
    self.write_metadata()

    self.encoder = None
    self.broken  = False
    self.empty   = True
    
  @property
  def closed(self) -> bool:
    """Return True, if disabled"""
    return not self.enabled

  @property
  def functional(self) -> bool:
    return self.enabled and not self.broken

  def capture_frame(self):
    if not self.functional:
      return
    # capture current frame
    frame = self.env.render(mode='rgb_array', **self._render_kwargs)
    if frame is None:
      if self._async:
        return
      else:
        logging.error("Env returned None on render(). Disabling further "
          f"rendering for video recorder by marking as disabled: {self.path}")
        self.broken = True
    else:
      self._encode_image_frame(frame)

  def close(self):
    if not self.enabled:
      return

    if self.encoder:
      #LOG.debug(f'Closing video encoder: {self.path}')
      self.encoder.close()
      self.encoder = None
    else:
      if os.path.isfile(self.path):
        os.remove(self.path)
      
      self.metadata['empty'] = True

    if self.broken:
      logging.warning(f"Cleaning up paths for broken video recorder: {self.path}")

      if os.path.isfile(self.path):
        os.remove(self.path)

      self.metadata['broken'] = True
    self.write_metadata()
    # mark as closed
    self.enabled = False

  def write_metadata(self):
    if not self.enabled:
      return
    try:
      with open(self.meta_path, 'wt') as f:
        json.dump(self.metadata, f, indent=2)
    except FileNotFoundError:
      logging.error(f"Failed to write metadata to path: {self.meta_path}\n"
        "This error may caused by trying to write files in a __del__ function.")

  def _create_path(
    self,
    dirpath: Optional[str] = None,
    filepath: Optional[str] = None
  ):
    """Safely create directories
    
    Args:
      dirpath (str, optional): directory path. Defaults to None.
      filepath (str, optional): file path. Defaults to None.
    """
    rl_utils.safe_makedirs(dirpath=dirpath, filepath=filepath)

  def _get_frame_shape(self, frame: np.ndarray) -> Tuple[int,int,int]:
    if len(frame.shape) != 3:
      raise RuntimeError(f"Receive unknown frame shape: {frame.shape}")
    return frame.shape # h,w,c

  def _get_video_shape(self, frame: np.ndarray) -> Tuple[int,int,int]:
    if len(frame.shape) != 3:
      raise RuntimeError(f"Receive unknown frame shape: {frame.shape}")
    h,w,c = frame.shape
    self.width = self.width if self.width else w
    self.height = self.height if self.height else h
    return (self.height, self.width, c) # h,w,c

  def _encode_image_frame(self, frame: np.ndarray):
    if not self.encoder:
      frame_shape = self._get_frame_shape(frame)
      video_shape = self._get_video_shape(frame)
      self.encoder = _VideoEncoder(
        self.path,
        frame_shape,
        video_shape,
        self.in_fps,
        self.out_fps
      )
      self.metadata['encoder_version'] = self.encoder.version_info
      self.metadata['video_info'] = self.encoder.video_info
    try:
      self.encoder.capture_frame(frame)
    except RuntimeError as e:
      logging.exception("Tried to pass invalid video frame, "
        f"marking as broken: {self.path}")
      self.metadata['exc'] = str(e)
      self.broken = True
    else:
      self.empty = False

  def __del__(self):
    if self.enabled:
      log_message(
        "The _VideoRecorder is not closed, please "
        "ensure you have closed the environment before the "
        "program ended `env.close()`",
        level="WARNING")
      self.close()


class VideoRecorder(MonitorToolChain):
  video_path: str = 'videos/'
  video_ext: str = 'video.mp4'
  metadata_ext: str = 'metadata.json'
  video_suffix: str = 'ep{episodes:06d}.{start_steps}-{steps}'
  def __init__(
    self,
    root_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    fps: Optional[int] = None,
    env_fps: Optional[int] = None,
    interval: Optional[Union[int, Callable]] = None,
    metadata: Optional[dict] = None,
    force: Optional[bool] = True
  ):
    """This monitor tool records the environment screens.

    Args:
      root_dir (str, optional): Root path to store video recordings. 
        Defaults to None.
      prefix (str, optional): Video recording file prefix. Defaults 
        to None.
      width (int, optional): Video record width. Defaults to None.
      height (int, optional): Video record height. Defaults to None.
      fps (int, optional): Video record FPS. Defaults to None.
      env_fps (int, optional): Environment's FPS. Defaults to None.
      interval (int, optional): Record interval (episodes). It can be
        an int or a callable function which has a signature:
          `def func(episode_number: int) -> bool`
        Return True if you want to record this number of episode,
        False otherwise. None for built in capped cubic video 
        schedule. Defaults to None.
      metadata (dict, optional): Metadata to write in additional json 
        file. Defaults to None.
      force (bool, optional): Force to save recordings. If False,
        it raises an exception if root_dir is not empty. Defaults 
        to True.

    Raises:
      RuntimeError: `interval` is not a callable object.
    """
    super().__init__()

    # Default callback
    if interval is None:
      schedule = VideoRecorder.capped_cubic_schedule
    elif isinstance(interval, int):
      schedule = lambda ep: (ep+1)%interval==0
    elif isinstance(interval, bool):
      schedule = lambda ep: interval
    elif not callable(interval):
      raise RuntimeError("You must provide a function or int for "
        f"`interval` not {type(interval)}: {interval}")

    self._root_dir  = root_dir
    self._prefix    = prefix
    self._suffix    = self.video_suffix
    self._video_ext = self.video_ext
    self._width     = width
    self._height    = height
    self._fps       = fps
    self._env_fps   = env_fps
    self._meta_ext  = self.metadata_ext
    self._metadata  = metadata
    self._schedule  = schedule
    self._force     = force

    self._is_reset  = False # Whether the env is reset
    self._env       = None
    self._enabled   = True
    self._recorder  = None
    self._monitor   = None
    self._stats     = None

    # initialized in `set_monitor`

  def set_monitor(self, monitor: Monitor):
    self._monitor = monitor
    self._stats   = monitor.stats
    self._env     = monitor.env
    # Default save root: {monitor.root_dir}/video/
    self._root_dir = (self._root_dir
      or os.path.join(self._monitor.root_dir, self.video_path))
    # Default prefix: {monitor.prefix}
    self._prefix = (self._prefix or self._monitor.prefix)
    # Raise an exception if the path is not empty
    if not self._force:
      if not self._path_is_empty(self._root_dir):
        raise RuntimeError("Try to write to non-empty video "
          f"directory {self._root_dir}")
  
  @staticmethod
  def capped_cubic_schedule(episode: int) -> bool:
    if episode < 1000:
      return int(round(episode ** (1./3))) ** 3 == episode
    else:
      return episode % 1000 == 0

  @property
  def need_record(self) -> bool:
    """Whether we should record this episode"""
    return self._schedule(self.stats.episodes)

  def close(self):
    """Close VideoRecorder"""
    if not self._enabled:
      return
    self._close_and_save_recorder()
    # disable VideoRecorder
    self._enabled = False

  def _before_step(self, act: Any) -> Any:
    if not self._enabled:
      return act

    # If the env is reset, create a new recorder
    if self._is_reset:
      self._new_recorder()
      # capture the first frame
      self._recorder.capture_frame()
      self._is_reset = False
    return act

  def _after_step(self, *data) -> Tuple[Any]:
    if not self._enabled:
      return data
    self._recorder.capture_frame()
    return data

  def _before_reset(self, **kwargs) -> Dict:
    if not self._enabled:
      return kwargs
    self._close_and_save_recorder()
    return kwargs

  def _after_reset(self, obs: Any) -> Any:
    if not self._enabled:
      return obs
    self._is_reset = True
    return obs

  def _create_path(
    self,
    dirpath: Optional[str] = None,
    filepath: Optional[str] = None
  ):
    """Safely create directories
    
    Args:
      dirpath (str, optional): directory path. Defaults to None.
      filepath (str, optional): file path. Defaults to None.
    """
    rl_utils.safe_makedirs(dirpath=dirpath, filepath=filepath)

  def _get_temp_path(
    self,
    dir: Optional[str],
    prefix: Optional[str],
    suffix: Optional[str]
  ) -> str:
    with tempfile.NamedTemporaryFile(dir=dir, prefix=prefix,
        suffix=suffix, delete=True) as f:
      path = str(f.name)
    return path

  def _make_path(
    self,
    root_dir: str,
    prefix: Optional[str],
    suffix: Optional[str],
    ext: str
  ) -> str:
    paths = []
    if prefix:
      paths.append(str(prefix))
    if suffix:
      paths.append(str(suffix))
    paths.append(str(ext))

    filename = '.'.join(paths)
    path = os.path.join(root_dir, filename)
    abspath = os.path.abspath(path)
    return abspath.format(**self.stats)

  def _close_and_save_recorder(self):
    if not self._recorder:
      return
    if not self._recorder.functional:
      self._recorder.close()
      return

    video_path = self._make_path(
      root_dir = self._root_dir,
      prefix   = self._prefix,
      suffix   = self._suffix,
      ext      = self._video_ext
    )
    meta_path  = self._make_path(
      root_dir = self._root_dir,
      prefix   = self._prefix,
      suffix   = self._suffix,
      ext      = self._meta_ext
    )

    video_relpath = os.path.relpath(video_path)
    meta_relpath  = os.path.relpath(meta_path)

    # update metadata
    episode_metadata = {
      'episode':         self.stats.episodes,
      'start_steps':     self.stats.start_steps,
      'end_steps':       self.stats.steps,
      'episode_length':  self.stats.steps - self.stats.start_steps,
      'episode_rewards': self.stats.rewards
    }

    self._recorder.metadata['episode_info'] = episode_metadata
    self._recorder.write_metadata()
    # close recorder after writintg metadata
    self._recorder.close()
    # create save paths
    self._create_path(filepath=video_path)
    self._create_path(filepath=meta_path)

    # copy tempfiles to the specified locations
    if os.path.isfile(self._recorder.meta_path):
      os.rename(self._recorder.meta_path, meta_path)
      #LOG.debug(f'Saving metadata to: {meta_path}')
    else:
      logging.warning(f"Metadata not found: {meta_path}")

    if os.path.isfile(self._recorder.path):
      os.rename(self._recorder.path, video_path)
      #LOG.debug(f'Saving video to: {video_path}')
    else:
      if self._recorder.broken:
        logging.warning("Failed to save video, the VideoRecorder is broken, "
          f"for more info: {meta_path}")
      else:
        logging.error("Failed to save video, missing tempfile, "
          f"for more info: {meta_path}")

  def _new_recorder(self):
    """Create a new video recorder"""

    temppath = self._make_path(
      root_dir = self._root_dir,
      prefix   = self._prefix,
      suffix   = None,
      ext      = 'ep{episodes:06d}.'
    )

    root_dir   = os.path.dirname(temppath)
    prefix     = os.path.basename(temppath)
    video_path = None
    meta_path  = None

    if self.need_record:
      self._create_path(root_dir)
      video_path = self._get_temp_path(root_dir, prefix, '.mp4')
      meta_path  = self._get_temp_path(root_dir, prefix, '.json')

    self._recorder = _VideoRecorder(
      env       = self._env,
      path      = video_path,
      meta_path = meta_path,
      metadata  = self._metadata,
      width     = self._width,
      height    = self._height,
      in_fps    = self._env_fps,
      out_fps   = self._fps,
      enabled   = self.need_record
    )
  
  def _path_is_empty(self, path: str) -> bool:
    if not path:
      return True
    return ((not os.path.isdir(path))
      or (len(os.listdir(path)) == 0))


# === Trajectory recorder ===


class _TrajectoryRecorder(object):
  def __init__(
    self,
    path: str,
    meta_path: str,
    metadata: Optional[dict] = None,
    enabled: bool = True
  ):
    """TrajectoryRecorder records trajectories
    from `env` and saves to `path`

    Args:
      path (str): trajectory save path
      meta_path (str): metadata save path
      metadata (dict, optional): metadata. Defaults to None.
      enabled (bool, optional): whether to enable this recorder.
        Defaults to True.
    """
    self.path = path
    self.meta_path = meta_path
    self.metadata = metadata or {}
    self.enabled = enabled
    if not self.enabled:
      return

    if not path:
      raise ValueError(f"`path` not specified: {path}")
    if not meta_path:
      raise ValueError(f"`meta_path` not specified: {meta_path}")

    # Create directory
    self._create_path(filepath=os.path.abspath(self.path))
    self._create_path(filepath=os.path.abspath(self.meta_path))
    # Write metadata
    self.metadata['content_type'] = 'application/octet-stream'
    self.write_metadata()

    self.empty = True
    self.buf = []

  @property
  def closed(self) -> bool:
    """Return True, if disabled"""
    return not self.enabled
  
  def close(self):
    if not self.enabled:
      return
    if not self.empty:
      self._write_trajectory()
    else:
      # clean up if it's empty
      if os.path.isfile(self.path):
        os.remove(self.path)
      self.update_trajectory_info_metadata(None, 0)
    self.write_metadata()
    self.enabled = False
    # free memory, gc
    del self.buf
    self.buf = []
    gc.collect()

  def cache_metadata(self, **metadata):
    """Cached metadata and save them to metadata file"""
    if not self.enabled:
      return
    metadata = copy.deepcopy(metadata)
    self.metadata.update(metadata)
    # update metadata
    self.write_metadata()

  def record_transition(self, info: Optional[Any]=None, **transition):
    """Record one transition."""
    if not self.enabled:
      return
    transition_data = dict(
      transition=transition,
      info=info
    )
    transition_data = copy.deepcopy(transition_data)
    self.buf.append(transition_data)
    self.empty = False

  def write_metadata(self):
    if not self.enabled:
      return
    try:
      rl_utils.safe_json_dump(self.meta_path, self.metadata, indent=2)
    except FileNotFoundError:
      logging.error(f"Failed to write metadata to path: {self.meta_path}\n"
        "This error may caused by trying to write files in a __del__ function.")

  def _create_path(
    self,
    dirpath: Optional[str] = None,
    filepath: Optional[str] = None
  ):
    """Safely create directories
    
    Args:
      dirpath (str, optional): directory path. Defaults to None.
      filepath (str, optional): file path. Defaults to None.
    """
    rl_utils.safe_makedirs(dirpath=dirpath, filepath=filepath)

  def _write_trajectory(self):
    """Save trajectory data (stored in self.buf) to
    the file
    """
    def data_to_numpy(data: Union[Tuple, List]) -> np.ndarray:
      """Combine all data in a list to one numpy array"""
      # TODO: ensure all the data having the same shape
      # and dtype. If not, convert it to a object array
      # and print a warning message.
      assert isinstance(data, (tuple, list))
      return np.asarray(data)
    # combine all dicts
    buf = rl_utils.flatten_dicts(self.buf)
    # process transitions
    num_transitions = len(buf['transition'])
    data = tuple(buf['transition'])
    data = rl_utils.map_nested_tuple(data, data_to_numpy)
    assert isinstance(data, dict)
    # process info, convert to object array
    info = np.asarray(buf['info'], dtype=object)
    data['info'] = info

    save_trajectory(self.path, data)
    self.update_trajectory_info_metadata(
      self.path,
      num_transitions
    )

  def update_trajectory_info_metadata(
    self,
    saved_path: str,
    num_transitions: int,
  ):
    """Update trajectory info metadata"""
    metadata = {
      'saved_path': saved_path,
      'num_transitions': num_transitions,
      'empty': num_transitions <= 0
    }
    self.metadata.setdefault('trajectory_info', {})
    # override metadata
    self.metadata['trajectory_info'].update(metadata)

  def __del__(self):
    if self.enabled:
      log_message(
        "WARNING: The _TrajectoryRecorder is not closed, please "
        "ensure you have closed the environment before the "
        "program ended `env.close()`",
        level="WARNING"
      )
      self.close()

class TrajectoryRecorder(MonitorToolChain):
  trajectory_path: str = 'trajs/'
  trajectory_ext: str = 'trajectory.npz'
  metadata_ext: str = 'metadata.json'
  trajectory_suffix: str = 'ep{episodes:06d}.{start_steps}-{steps}'
  def __init__(
    self,
    root_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    interval: Optional[Union[int, Callable]] = None,
    metadata: Optional[dict] = None,
    force: bool = True,
    enabled: bool = True
  ):
    """This monitor tool records the environment trajectories and save
    them as a npz file. The trajectory files are saved to
    ```
    {root_dir}/{prefix}.{trajectory_suffix}.{trajectory_ext}
    ```
    The trajectory metadata are saved to
    ```
    {root_dir}/{prefix}.{trajectory_suffix}.{metadata_ext}
    ```
    TODO: Note that the trajectory metadata may overwrite the video
    recording metadata, if both are saved to the same directory. The
    video recording metadata and trajectory metadata may be joined
    in the future version.

    The recording `interval` can be an int, bool or a callable function.
    An `int` indicates to record once for every `interval` episodes. For
    a `bool`, `True` to record all episodes, `False` to disable the
    recorder. The callable function should have the following signature:
    ```python
    def func(episode_num: int) -> bool:
      return True
    ```
    The function returns `True` if you want to record this episode, `False`
    otherwise. Set to `None` for built-in capped cubic video schedule,
    which records 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000-th episode
    when `episode_number` is less than 1000. And records every 1000-th
    episode, e.g. 1000, 2000, 3000..., when its beyond 1000.

    Args:
      root_dir (str, optional): root path to store trajectories.
        Defaults to None.
      prefix (str, optional): trajectory file prefix. Defaults
        to None.
      interval (int, bool, optional): record interval (episodes).
        Defaults to None.
      metadata (dict, optional): Metadata to write in additional json 
        file. Defaults to None.
      force (bool, optional): Force to save recordings. If False,
        it raises an exception if root_dir is not empty. Defaults 
        to True.
      enabled (bool, optional): enable this trajectory recorder.

    Raises:
      RuntimeError: invalid `interval`.
    """
    super().__init__()
    # Default callback
    if interval is None:
      schedule = TrajectoryRecorder.capped_cubic_schedule
    elif isinstance(interval, int):
      schedule = lambda ep: (ep+1)%interval==0
    elif isinstance(interval, bool):
      schedule = lambda ep: interval
    elif not callable(interval):
      raise RuntimeError("You must provide an int/bool/callable for "
        f"`interval` not {type(interval)}: {interval}")
    
    self._root_dir = root_dir
    self._prefix = prefix
    self._suffix = self.trajectory_suffix
    self._traj_ext = self.trajectory_ext
    self._meta_ext = self.metadata_ext
    self._metadata = metadata
    self._schedule = schedule
    self._force = force

    self._is_reset = False
    self._env = None
    self._enabled = enabled
    self._monitor = None
    self._stats = None
    self._recorder = None
    # caches
    self._cached_metadata = None
    self._cached_observation = None
    self._cached_action = None

  def set_monitor(self, monitor: Monitor):
    self._monitor = monitor
    self._stats = monitor.stats
    self._env = monitor.env
    # Default save root: {monitor.root_dir}/traj/
    self._root_dir = (self._root_dir
      or os.path.join(self._monitor.root_dir, self.trajectory_path))
    # Default prefix: {monitor.prefix}
    self._prefix = (self._prefix or self._monitor.prefix)
    # Raise an exception if the path is not empty
    if not self._force:
      if not self._path_is_empty(self._root_dir):
        raise RuntimeError("Try to write to non-empty trajectory "
          f"directory {self._root_dir}")
    
  @staticmethod
  def capped_cubic_schedule(episode: int) -> bool:
    if episode < 1000:
      return int(round(episode ** (1./3))) ** 3 == episode
    else:
      return episode % 1000 == 0

  @property
  def need_record(self) -> bool:
    """Whether we should record this episode"""
    return self._schedule(self.stats.episodes)
  
  def close(self):
    """Close TrajectoryRecorder"""
    if not self._enabled:
      return
    self._close_and_save_recorder()
    # disable TrajectoryRecorder
    self._enabled = False

  def _before_step(self, act: Any) -> Any:
    if not self._enabled:
      return act
    self._cached_action = copy.deepcopy(act)
    return act

  def _after_step(self, obs: Any, rew: Any, done: Any, info: Any) -> Tuple[Any]:
    if not self._enabled:
      return (obs, rew, done, info)
    self._recorder.record_transition(
      obs = self._cached_observation,
      act = self._cached_action,
      next_obs = obs,
      rew = rew,
      done = done,
      info = info
    )
    self._cached_observation = copy.deepcopy(obs)
    return (obs, rew, done, info)

  def _before_reset(self, **kwargs) -> Dict:
    if not self._enabled:
      return kwargs
    self._close_and_save_recorder()
    self._cached_metadata = dict(
      reset_options = copy.deepcopy(kwargs)
    )
    return kwargs

  def _after_reset(self, obs: Any) -> Any:
    if not self._enabled:
      return obs
    self._new_recorder()
    self._recorder.cache_metadata(**self._cached_metadata)
    self._cached_observation = copy.deepcopy(obs)
    return obs

  def _create_path(
    self,
    dirpath: Optional[str] = None,
    filepath: Optional[str] = None
  ):
    """Safely create directories
    
    Args:
      dirpath (str, optional): directory path. Defaults to None.
      filepath (str, optional): file path. Defaults to None.
    """
    rl_utils.safe_makedirs(dirpath=dirpath, filepath=filepath)
  
  def _get_temp_path(
    self,
    dir: Optional[str],
    prefix: Optional[str],
    suffix: Optional[str]
  ) -> str:
    with tempfile.NamedTemporaryFile(dir=dir, prefix=prefix,
        suffix=suffix, delete=True) as f:
      path = str(f.name)
    return path

  def _make_path(
    self,
    root_dir: str,
    prefix: Optional[str],
    suffix: Optional[str],
    ext: str
  ) -> str:
    paths = []
    if prefix:
      paths.append(str(prefix))
    if suffix:
      paths.append(str(suffix))
    paths.append(str(ext))

    filename = '.'.join(paths)
    path = os.path.join(root_dir, filename)
    abspath = os.path.abspath(path)
    return abspath.format(**self.stats)

  def _close_and_save_recorder(self):
    """Close and save the recorder. This function is expected to be called
    on resetting or closing the environment.
    """
    if not self._recorder:
      return
    if self._recorder.closed:
      return

    traj_path = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = self._suffix,
      ext = self._traj_ext
    )
    meta_path = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = self._suffix,
      ext = self._meta_ext
    )
    traj_relpath = os.path.relpath(traj_path)
    meta_relpath = os.path.relpath(meta_path)
    # update metadata
    episode_metadata = {
      'episode': self.stats.episodes,
      'start_steps': self.stats.start_steps,
      'end_steps': self.stats.steps,
      'episode_length': self.stats.steps - self.stats.start_steps,
      'episode_rewards': self.stats.rewards
    }
    self._recorder.metadata.setdefault('episode_info', {})
    self._recorder.metadata['episode_info'].update(episode_metadata)
    trajinfo_metadata = {
      'path': traj_path,
      'relpath': os.path.relpath(
        os.path.abspath(traj_path),
        os.path.dirname(os.path.abspath(meta_path))
      )
    }
    self._recorder.metadata.setdefault('trajectory_info', {})
    self._recorder.metadata['trajectory_info'].update(trajinfo_metadata)
    self._recorder.write_metadata()
    # close recorder before moving saved files
    # Note that the trajectory data will be cleaned after `close()`
    self._recorder.close()
    self._create_path(filepath=traj_path)
    self._create_path(filepath=meta_path)
    # copy temp metadata to the specified locations
    if os.path.isfile(self._recorder.meta_path):
      os.rename(self._recorder.meta_path, meta_path)
      #LOG.debug(f'Saving trajectory metadata to: {meta_path}')
    else:
      logging.warning(f"Metadata not found: {meta_path}")
    # copy temp traj file to the specified locations if it is saved
    if self._recorder.empty:
      return
    if os.path.isfile(self._recorder.path):
      os.rename(self._recorder.path, traj_path)
      #LOG.debug(f'Saving trajectory to: {traj_path}')
    else:
      logging.warning(f"Failed to save trajectory, for more info: {meta_path}")

  def _new_recorder(self):
    """Create a new trajectory recorder"""
    # The temppath is a temporary location which has the form
    # {prefix}.{ext}{tempfile}{suffix}
    # the final `.` for `ext` is used to split the {ext} and {tempfile}
    temppath = self._make_path(
      root_dir = self._root_dir,
      prefix = self._prefix,
      suffix = None,
      ext = 'ep{episodes:06d}.'
    )
    # normalize save path, the `prefix` should not contain any `/`
    root_dir = os.path.dirname(temppath)
    prefix = os.path.basename(temppath)
    traj_path = None
    meta_path = None
    # make paths
    if self.need_record:
      self._create_path(root_dir)
      traj_path = self._get_temp_path(root_dir, prefix, '.npz')
      meta_path = self._get_temp_path(root_dir, prefix, '.json')
    # create trajectory recorder
    # the recorder should be disabled if the path is empty
    # (only occurred if we don't need to record this episode)
    # otherwise, a ValueError is raised.
    self._recorder = _TrajectoryRecorder(
      path = traj_path,
      meta_path = meta_path,
      metadata = self._metadata,
      enabled = self.need_record
    )

  def _path_is_empty(self, path: str) -> bool:
    """Return True if the path (dir) does not exist,
    or it's empty.
    """
    if not path:
      return True
    return ((not os.path.isdir(path))
      or (len(os.listdir(path)) == 0))
