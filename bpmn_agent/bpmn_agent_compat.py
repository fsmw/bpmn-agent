"""Compatibility module for imports."""

import sys
from pathlib import Path

# Add current directory to path so modules can be imported as bpmn_agent.*
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Create a virtual bpmn_agent module
import importlib  # noqa: E402
import types  # noqa: E402

bpmn_agent = types.ModuleType("bpmn_agent")

# Import all submodules and add them to bpmn_agent
submodules = ["core", "knowledge", "models", "stages", "tools", "validators"]
for submodule in submodules:
    try:
        mod = importlib.import_module(submodule)
        setattr(bpmn_agent, submodule, mod)
    except ImportError:
        pass

sys.modules["bpmn_agent"] = bpmn_agent
