"""
Ensure that Python commands use the local virtual environment.
This script must be imported at the start of any Python execution.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PATH = PROJECT_ROOT / ".venv" / "bin" / "python"

# Check if we're using the local venv
if sys.prefix != str(PROJECT_ROOT / ".venv"):
    print(f"❌ ERROR: Not using local virtual environment!")
    print(f"   Current Python: {sys.executable}")
    print(f"   Expected: {VENV_PATH}")
    print(f"\nFIX: Use the local venv:")
    print(f"   {VENV_PATH} <your_script>.py")
    sys.exit(1)

# Ensure PYTHONPATH includes the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"✓ Using local venv: {sys.executable}")
