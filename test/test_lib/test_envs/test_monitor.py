# --- built in ---
import os
import json
import tempfile
# --- 3rd party --
import numpy as np
# --- my module ---
from rlchemy.lib import utils as rl_utils
from rlchemy.lib import envs as rl_envs
from test.utils import TestCase
from test.test_lib.test_envs.utils import FakeImageEnv, FakeContinuousEnv

class NoopMonitorTool(rl_envs.MonitorToolChain):
  def __init__(self):
    super().__init__()


class BrokenImageEnv(FakeImageEnv):
  def render(self, mode='human'):
    return None

class NonRenderableImageEnv(FakeImageEnv):
  metadata = {'render.modes': []}

class TestMonitorModule(TestCase):
  """Test rlchemy.lib.envs.monitor module"""
  def test_monitor_wo_video_recorder(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_wo_video_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')

      env = rl_envs.Monitor(env, root_dir=root_dir, video=False)
      self.assertEqual(1, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(env.stats is not None)
      # assert csv created
      csv_path = os.path.join(root_dir, rl_envs.StatsRecorder.monitor_ext)
      self.assertTrue(os.path.isfile(csv_path))
      # test add tools
      env.add_tools([NoopMonitorTool(), NoopMonitorTool()])
      self.assertEqual(3, len(env.tools))
      env.reset()
      for i in range(25):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
          env.reset()
      env.reset() # early reset
      with open(csv_path,'r') as f:
        lines = f.readlines()
      self.assertEqual(5, len(lines))
      self.assertTrue('"env_id": "FakeEnv"' in lines[0])
      self.assertTrue('rewards,length,walltime' in lines[1])
      self.assertTrue('55,10' in lines[2])
      self.assertTrue('55,10' in lines[3])
      self.assertTrue('15,5' in lines[4]) # early reset episode
      env.close()
      self.assertTrue(env.tools[0].closed)
  
  def test_monitor_wo_video_recorder_exception(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_wo_video_recorder_exception')
      root_dir = os.path.join(root_dir, 'monitor/')

      env = rl_envs.Monitor(env, root_dir=root_dir, video=False)
      with self.assertRaises(RuntimeError):
        # test exception: need reset
        for i in range(20):
          env.step(env.action_space.sample())
      env.close()
      self.assertTrue(env.tools[0].closed)

  def test_video_recorder_capped_cubic_schedule(self):
    for i in range(1, 2000+1):
      res = rl_envs.VideoRecorder.capped_cubic_schedule(i)
      if i in [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 2000]:
        self.assertTrue(res)
      else:
        self.assertFalse(res)

  def test_monitor_w_video_recorder(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_w_video_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      
      for ep in range(10):
        env.reset()
        self.assertTrue(env.tools[1].need_record)
        for step in range(20):
          obs, rew, done, info = env.step(env.action_space.sample())
          if done:
            break
      env.close()
      video_path = rl_envs.VideoRecorder.video_path
      video_path = os.path.join(root_dir, video_path)
      files = sorted(os.listdir(video_path))
      self.assertEqual(10*2, len(files)) # json + mp4
      for filename in files:
        self.assertTrue(
          filename.endswith('metadata.json') or
          filename.endswith('video.mp4')
        , filename)
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)
      json_files = [f for f in files if f.endswith('.json')]
      json_files[0]
      with open(os.path.join(video_path, json_files[0]), 'r') as f:
        meta = json.loads(f.read())
      self.assertTrue('episode_info' in meta)
      self.assertTrue('video_info' in meta)
      self.assertTrue('encoder_version' in meta)
      self.assertTrue('content_type' in meta)
  
  def test_monitor_w_video_recorder_cubic(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_w_video_recorder_cubic')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=None)
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      
      for ep in range(10):
        env.reset()
        self.assertEqual(ep+1 in [1,8], env.tools[1].need_record)
        for step in range(20):
          obs, rew, done, info = env.step(env.action_space.sample())
          if done:
            break
      env.close()
      video_path = rl_envs.VideoRecorder.video_path
      video_path = os.path.join(root_dir, video_path)
      records = sorted(os.listdir(video_path))
      self.assertEqual(2*2, len(records)) # json + mp4
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)
  
  def test_video_recorder_init_exception(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_video_recorder_init_exception')
      root_dir = os.path.join(root_dir, 'monitor/')
      with self.assertRaises(RuntimeError):
        # interval not callable
        env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval='hello'))
      env.close()
      # TODO: dangling StatsRecorder

  def test_monitor_stats_recorder_video_recorder_prefix(self):
    prefix = 'eval'
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_stats_recorder_video_recorder_prefix')
      root_dir = os.path.join(root_dir, 'monitor/')
      env = rl_envs.Monitor(env, root_dir=root_dir, prefix=prefix,
                video=True, video_kwargs=None)
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      env.reset()
      for i in range(25):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
          env.reset()
      csv_path = os.path.join(root_dir, 
            prefix + '.' + rl_envs.StatsRecorder.monitor_ext)
      self.assertTrue(os.path.isfile(csv_path))
      video_path = rl_envs.VideoRecorder.video_path
      video_path = os.path.join(root_dir, video_path)
      for filename in os.listdir(video_path):
        self.assertTrue(filename.startswith(prefix))
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)
  
  def test_monitor_broken_image_env(self):
    env = BrokenImageEnv(max_steps=10)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_broken_image_env')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      env.reset()
      # The brokwn image env returns None when capture the frame
      print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Test broken image env')
      env.step(env.action_space.sample())
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
      self.assertTrue(env.tools[1]._recorder.broken)
      self.assertFalse(env.tools[1]._recorder.functional)
      for i in range(10):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
          break
      # show log: VideoRecorder is broken
      env.reset()
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)

  def test_monitor_non_recordable_image_env(self):
    env = NonRenderableImageEnv(max_steps=10)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_non_recordable_image_env')
      root_dir = os.path.join(root_dir, 'monitor/')
      video_path = rl_envs.VideoRecorder.video_path
      video_path = os.path.join(root_dir, video_path)
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))

      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      env.reset()
      # The non-recordable image env has no 'rgb_array' mode
      print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Test non-recordable image env')
      env.step(env.action_space.sample())
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
      self.assertFalse(env.tools[1]._recorder.enabled)
      self.assertFalse(env.tools[1]._recorder.functional)
      for i in range(10):
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
          break
      env.reset()
      env.close()
      self.assertEqual(0, len(os.listdir(video_path)))
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)

  def test_monitor_non_empty_folder(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_non_empty_folder')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      
      for ep in range(10):
        env.reset()
        self.assertTrue(env.tools[1].need_record)
        for step in range(20):
          obs, rew, done, info = env.step(env.action_space.sample())
          if done:
            break
      env.close()
      video_path = rl_envs.VideoRecorder.video_path
      video_path = os.path.join(root_dir, video_path)
      records = sorted(os.listdir(video_path))
      self.assertEqual(10*2, len(records)) # json + mp4
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)
      # writing monitor to a non empty file with force=False raises
      # a RuntimeError
      env = FakeImageEnv(max_steps=10, channel_first=False)
      with self.assertRaises(RuntimeError):
        rl_envs.Monitor(env, root_dir=root_dir, video=True,
              video_kwargs=dict(interval=1, force=False))
      env.close()
    
  def test_monitor_disabled_video_recorder(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_disabled_video_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      # _VideoRecorder will not be created
      env.tools[1].close()
      
      for ep in range(10):
        env.reset()
        self.assertTrue(env.tools[1].need_record)
        for step in range(20):
          obs, rew, done, info = env.step(env.action_space.sample())
          if done:
            break
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)

  def test_video_recorder_invalid_frame(self):
    env = FakeImageEnv(max_steps=10, channel_first=False)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_video_recorder_invalid_frame')
      root_dir = os.path.join(root_dir, 'monitor/')
      # always record
      env = rl_envs.Monitor(env, root_dir=root_dir, video=True,
                video_kwargs=dict(interval=1))
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.VideoRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      env.reset()
      # start video encoder
      env.step(env.action_space.sample())
      # test invalid frame
      print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Test invalid frame type np.float32')
      env.tools[1]._recorder._encode_image_frame(
        np.random.normal(size=(64, 64, 3)).astype(np.float32)
      )
      print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
      self.assertTrue(env.tools[1]._recorder.broken)
      with self.assertRaises(RuntimeError):
        # Not a np.ndarray
        env.tools[1]._recorder.encoder.capture_frame(
          'foobar'
        )
      with self.assertRaises(RuntimeError):
        env.tools[1]._recorder.encoder.capture_frame(
          np.zeros((64, 32, 3), dtype=np.uint8)
        )
      # Test statsrecorder, videorecorder
      # auto closed
      self.assertFalse(env.tools[0].closed)
      self.assertTrue(env.tools[1]._enabled)
      self.assertFalse(env.tools[1]._recorder.closed)

  def test_trajectory_recorder_capped_cubic_schedule(self):
    for i in range(1, 2000+1):
      res = rl_envs.TrajectoryRecorder.capped_cubic_schedule(i)
      if i in [0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 2000]:
        self.assertTrue(res)
      else:
        self.assertFalse(res)

  def test_monitor_w_trajectory_recorder(self):
    env = FakeContinuousEnv(max_steps=10)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_w_trajectory_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')
      env = rl_envs.Monitor(env, root_dir=root_dir, video=False)
      # add trajectory recorder to monitor tool chain
      env.add_tool(
        rl_envs.TrajectoryRecorder(interval=True, force=True)
      )
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.TrajectoryRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      
      for ep in range(10):
        buf = []
        obs = env.reset()
        self.assertTrue(env.tools[1].need_record)
        for step in range(20):
          act = env.action_space.sample()
          next_obs, rew, done, info = env.step(act)
          buf.append(dict(
            obs = obs,
            act = act,
            next_obs = next_obs,
            rew = rew,
            done = done,
            info = info
          ))
          obs = next_obs
          if done:
            break
      env.close()
      buf = rl_utils.flatten_dicts(buf)
      for key, value in buf.items():
        buf[key] = np.asarray(value)
      trajectory_path = rl_envs.TrajectoryRecorder.trajectory_path
      trajectory_path = os.path.join(root_dir, trajectory_path)
      files = sorted(os.listdir(trajectory_path))
      self.assertEqual(10*2, len(files)) # json + npz
      for filename in files:
        self.assertTrue(
          filename.endswith(rl_envs.TrajectoryRecorder.metadata_ext) or
          filename.endswith(rl_envs.TrajectoryRecorder.trajectory_ext)
        , filename)
      env.close()
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)
      json_files = [f for f in files if f.endswith('.json')]
      full_json_path = os.path.join(trajectory_path, json_files[-1])
      meta = rl_utils.safe_json_load(full_json_path)
      self.assertTrue('episode_info' in meta)
      self.assertTrue('trajectory_info' in meta)
      self.assertTrue('reset_options' in meta)
      self.assertTrue('content_type' in meta)
      self.assertTrue('path' in meta['trajectory_info'])
      self.assertTrue('relpath' in meta['trajectory_info'])
      self.assertTrue('saved_path' in meta['trajectory_info'])
      self.assertTrue('num_transitions' in meta['trajectory_info'])
      self.assertTrue('empty' in meta['trajectory_info'])
      self.assertEqual(10, meta['trajectory_info']['num_transitions'])
      self.assertFalse(meta['trajectory_info']['empty'])
      # check npz contents
      npz_files = [f for f in files if f.endswith('.npz')]
      full_npz_path = os.path.join(trajectory_path, npz_files[-1])
      contents = rl_envs.load_trajectory(full_npz_path)
      self.assertEqual(set(buf.keys()), set(contents.keys()))
      for key in buf.keys():
        if buf[key].dtype == np.object_:
          self.assertArrayEqual(buf[key], contents[key])
        else:
          self.assertArrayClose(buf[key], contents[key])

  def test_trajectory_recorder_init_exceptions(self):
    env = FakeContinuousEnv(max_steps=10)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_w_trajectory_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')
      env = rl_envs.Monitor(env, root_dir=root_dir, video=False)
      # test invalid interval
      with self.assertRaises(RuntimeError):
        env.add_tool(
          rl_envs.TrajectoryRecorder(interval='hello', force=True)
        )
      # test auto close
      env.add_tool(
        rl_envs.TrajectoryRecorder(interval=1, force=True)
      )
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.TrajectoryRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      for ep in range(1):
        obs = env.reset()
        self.assertTrue(env.tools[1].need_record)
        for step in range(20):
          act = env.action_space.sample()
          next_obs, rew, done, info = env.step(act)
          if done:
            break

  def test_trajectory_recorder_interval(self):
    env = FakeContinuousEnv(max_steps=10)
    with tempfile.TemporaryDirectory() as tempdir:
      root_dir = os.path.join(tempdir, 'test_monitor_w_trajectory_recorder')
      root_dir = os.path.join(root_dir, 'monitor/')
      env = rl_envs.Monitor(env, root_dir=root_dir, video=False)
      # add trajectory recorder to monitor tool chain
      env.add_tool(
        rl_envs.TrajectoryRecorder(interval=2, force=True)
      )
      self.assertEqual(2, len(env.tools))
      self.assertTrue(isinstance(env.tools[0], rl_envs.StatsRecorder))
      self.assertTrue(isinstance(env.tools[1], rl_envs.TrajectoryRecorder))
      self.assertTrue(env.tools[1].stats is not None)
      
      for ep in range(10):
        obs = env.reset()
        self.assertEqual((ep+1)%2, env.tools[1].need_record)
        for step in range(20):
          act = env.action_space.sample()
          next_obs, rew, done, info = env.step(act)
          if done:
            break
      env.close()
      trajectory_path = rl_envs.TrajectoryRecorder.trajectory_path
      trajectory_path = os.path.join(root_dir, trajectory_path)
      files = sorted(os.listdir(trajectory_path))
      self.assertEqual(5*2, len(files)) # json + npz
      for filename in files:
        self.assertTrue(
          filename.endswith(rl_envs.TrajectoryRecorder.metadata_ext) or
          filename.endswith(rl_envs.TrajectoryRecorder.trajectory_ext)
        , filename)
      self.assertTrue(env.tools[0].closed)
      self.assertFalse(env.tools[1]._enabled)
      self.assertTrue(env.tools[1]._recorder.closed)