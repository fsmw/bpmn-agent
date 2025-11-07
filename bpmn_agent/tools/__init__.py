"""
BPMN Agent Tools

Utilities and CLI tools for the BPMN Agent.
"""

try:
    from bpmn_agent.tools.cli import cli

    __all__ = ["cli"]
except (ImportError, SystemExit):
    # CLI not available in testing environment
    __all__ = []
