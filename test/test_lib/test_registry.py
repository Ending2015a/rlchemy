# --- built in ---
# --- 3rd party ---
# --- my module ---
from rlchemy.lib import registry as rl_registry
from test.utils import TestCase

class TestRegistryModule(TestCase):
  """Test rlchemy.lib.registry module"""
  def test_registry_singleton(self):
    self.assertTrue(rl_registry.Registry() is rl_registry.Registry())
  
  def test_registry(self):
    rl_registry.registry.reset()
    class A:
      pass
    class B:
      pass
    rl_registry.registry.register('a', 'A')(A)
    rl_registry.registry.register('b', 'B')(B)
    self.assertTrue(rl_registry.registry.get('a', 'A') == A)
    self.assertTrue(rl_registry.registry.get('b', 'B') == B)
    self.assertTrue(rl_registry.registry.get('a', 'B') is None)
    self.assertTrue(rl_registry.registry.get('b', 'A') is None)
    self.assertTrue(rl_registry.registry.get('a', 'B', 100) == 100)
  
  def test_registry_alias(self):
    rl_registry.registry.reset()
    class A:
      pass
    rl_registry.registry.register('a', ['A', 'AA', 'AAA'])(A)
    self.assertTrue(rl_registry.registry.get('a', 'A') == A)
    self.assertTrue(rl_registry.registry.get('a', 'AA') == A)
    self.assertTrue(rl_registry.registry.get('a', 'AAA') == A)
    self.assertTrue(rl_registry.registry.get('a', 'AAAA') is None)

  def test_registry_default(self):
    rl_registry.registry.reset()
    class A:
      pass
    class DefaultA:
      pass
    rl_registry.registry.register('a', ['A'])(A)
    rl_registry.registry.register('a', ['DefaultA'], default=True)(DefaultA)
    self.assertTrue(rl_registry.registry.get('a', 'A') == A)
    self.assertTrue(rl_registry.registry.get('a', 'DefaultA') == DefaultA)
    self.assertTrue(rl_registry.registry.get('a') == DefaultA)
    self.assertTrue(rl_registry.registry.get('a', 'default') == DefaultA)
  
  def test_registry_overwrite(self):
    rl_registry.registry.reset()
    class A:
      pass
    class OverwriteA:
      pass
    rl_registry.registry.register('a', ['A'])(A)
    self.assertTrue(rl_registry.registry.get('a', 'A') == A)
    with self.assertRaises(AssertionError):
      rl_registry.registry.register('a', ['A'])(OverwriteA)
    self.assertTrue(rl_registry.registry.get('a', 'A') == A)
    rl_registry.registry.register('a', ['A'], overwrite=True)(OverwriteA)
    self.assertTrue(rl_registry.registry.get('a', 'A') == OverwriteA)

  def test_register(self):
    rl_registry.registry.reset()
    @rl_registry.register('a', ['A'])
    class A:
      pass
    self.assertTrue(rl_registry.get('a', 'A') == A)
    self.assertTrue(rl_registry.get.a('A') == A)
    self.assertTrue(rl_registry.get.a('B', 100) == 100)

    @rl_registry.register.b(['B'])
    class B:
      pass
    self.assertTrue(rl_registry.get('b', 'B') == B)
    self.assertTrue(rl_registry.get.b('B') == B)
    self.assertTrue(rl_registry.get.b('A', 100) == 100)
