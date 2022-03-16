# --- built in ---
# --- 3rd party ---
import omegaconf
from omegaconf import OmegaConf
from parameterized import parameterized
# --- my module ---
from rlchemy.lib.conf import resolvers as rl_resolvers
from test.utils import TestCase

class TestConfResolversModule(TestCase):
  @parameterized.expand([
    (5,),
    (1,4),
    (0,8,2)
  ])
  def test_resolver_range(self, *args):
    str_args = ",".join([str(a) for a in args])
    exp = list(range(*args))
    conf = OmegaConf.create("a: ${range:" + str_args + "}")
    # convert to pure python container, expecting a list
    pyconf = OmegaConf.to_container(conf, resolve=True)
    print(pyconf)
    self.assertTrue(isinstance(pyconf['a'], list))
    self.assertArrayEqual(pyconf['a'], exp)
    # resolve in omegaconf container
    OmegaConf.resolve(conf)
    self.assertTrue(isinstance(conf.a, omegaconf.ListConfig))
    for i in range(len(exp)):
      self.assertEqual(conf.a[i], exp[i])


