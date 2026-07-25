"""Test-only stubs for optional runtime packages unavailable in CI."""

import sys
from types import ModuleType


try:
    import gym  # noqa: F401
except ModuleNotFoundError:
    gym_module = ModuleType("gym")
    envs_module = ModuleType("gym.envs")
    registration_module = ModuleType("gym.envs.registration")
    registration_module.register = lambda **_kwargs: None
    gym_module.envs = envs_module
    envs_module.registration = registration_module
    sys.modules["gym"] = gym_module
    sys.modules["gym.envs"] = envs_module
    sys.modules["gym.envs.registration"] = registration_module
