"""Pytest configuration for bpmn-agent tests."""

import sys
from pathlib import Path
import pytest

# Add the bpmn-agent source directory to the path
bpmn_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bpmn_agent_dir))

# Create a virtual bpmn_agent package that maps to the current directory
import types
bpmn_agent = types.ModuleType('bpmn_agent')
bpmn_agent.__path__ = [str(bpmn_agent_dir)]
sys.modules['bpmn_agent'] = bpmn_agent

# Dynamically add submodules to the namespace
import importlib
submodule_names = ['core', 'knowledge', 'models', 'stages', 'tools', 'validators']
for name in submodule_names:
    try:
        mod = importlib.import_module(name)
        setattr(bpmn_agent, name, mod)
        sys.modules[f'bpmn_agent.{name}'] = mod
    except Exception:
        pass

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """Configure pytest with asyncio support."""
    # Set asyncio mode to auto for pytest-asyncio
    config.option.asyncio_mode = 'auto'
