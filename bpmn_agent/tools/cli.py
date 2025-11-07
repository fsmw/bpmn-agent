"""
BPMN Agent CLI Interface

Command-line tool for direct agent usage, supporting various input formats
and output options.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# ENFORCE LOCAL VENV USAGE
_PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
_VENV_PATH = _PROJECT_ROOT / ".venv"
if sys.prefix != str(_VENV_PATH):
    click.echo("❌ ERROR: Not using local virtual environment!", err=True)
    click.echo(f"   Current: {sys.executable}", err=True)
    click.echo(f"   Expected: {_VENV_PATH / 'bin' / 'python'}", err=True)
    click.echo("\nFIX: Always use the local venv for this project:", err=True)
    click.echo(f"   {_VENV_PATH / 'bin' / 'python'} -m bpmn_agent.tools.cli [command]", err=True)
    sys.exit(1)

from bpmn_agent.agent import AgentConfig, BPMNAgent, ProcessingMode
from bpmn_agent.agent.state import AgentState
from bpmn_agent.core.llm_client import LLMConfig
from bpmn_agent.core.observability import LogLevel, ObservabilityConfig, ObservabilityManager

# Setup logging
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """BPMN Agent CLI - Transform natural language to BPMN diagrams."""
    pass


@cli.command()
@click.argument("input_text", required=False)
@click.option(
    "--input-file",
    "-f",
    type=click.Path(exists=True),
    help="Read input from file instead of command line argument",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for generated BPMN XML",
)
@click.option(
    "--process-name",
    "-n",
    type=str,
    help="Override process name in output",
)
@click.option(
    "--mode",
    type=click.Choice(["standard", "kb_enhanced", "analysis_only", "validation_only"]),
    default="standard",
    help="Processing mode",
)
@click.option(
    "--domain",
    "-d",
    type=str,
    help="Specify process domain (auto-detected if not provided)",
)
@click.option(
    "--llm-provider",
    type=click.Choice(["ollama", "openai_compatible"]),
    default="ollama",
    help="LLM provider",
)
@click.option(
    "--llm-base-url",
    default=None,
    help="LLM base URL",
)
@click.option(
    "--llm-model",
    default="mistral",
    help="LLM model name",
)
@click.option(
    "--llm-api-key",
    default=None,
    help="LLM API key (if required)",
)
@click.option(
    "--enable-kb/--disable-kb",
    default=True,
    help="Enable/disable knowledge base patterns",
)
@click.option(
    "--verbose/--quiet",
    default=False,
    help="Verbose logging output",
)
@click.option(
    "--json-output",
    is_flag=True,
    help="Output in JSON format (includes metadata)",
)
def process(
    input_text: Optional[str],
    input_file: Optional[str],
    output: Optional[str],
    process_name: Optional[str],
    mode: str,
    domain: Optional[str],
    llm_provider: str,
    llm_base_url: Optional[str],
    llm_model: str,
    llm_api_key: Optional[str],
    enable_kb: bool,
    verbose: bool,
    json_output: bool,
) -> None:
    """
    Process natural language text to generate BPMN XML.

    \b
    Examples:
        # Process text from command line
        bpmn-agent process "A customer submits an order..."

        # Process from file
        bpmn-agent process --input-file description.txt --output diagram.bpmn

        # Use KB-enhanced mode with domain
        bpmn-agent process -f process.txt -o out.xml --mode kb_enhanced --domain finance
    """

    # Setup observability
    if verbose:
        obs_config = ObservabilityConfig(
            service_name="bpmn-agent-cli",
            log_level=LogLevel.DEBUG,
            json_logs=json_output,
        )
    else:
        obs_config = ObservabilityConfig(
            service_name="bpmn-agent-cli",
            log_level=LogLevel.INFO,
            json_logs=json_output,
        )

    ObservabilityManager.initialize(obs_config)

    # Get input text
    if input_file:
        try:
            with open(input_file, "r") as f:
                text = f.read()
        except IOError as e:
            click.echo(f"Error reading input file: {e}", err=True)
            sys.exit(1)
    elif input_text:
        text = input_text
    else:
        # Read from stdin if no input provided
        click.echo("Reading from stdin (press Ctrl+D to finish)...", err=True)
        text = sys.stdin.read()

    if not text.strip():
        click.echo("Error: No input text provided", err=True)
        sys.exit(1)

    # Create agent configuration
    llm_config = LLMConfig(
        provider=llm_provider,
        base_url=llm_base_url or _get_default_llm_url(llm_provider),
        model=llm_model,
        api_key=llm_api_key,
    )

    agent_config = AgentConfig(
        llm_config=llm_config,
        mode=ProcessingMode(mode),
        enable_kb=enable_kb,
        enable_logging=True,
        verbose=verbose,
    )

    # Create and run agent
    agent = BPMNAgent(agent_config)

    try:
        xml, state = asyncio.run(agent.process(text, process_name=process_name, domain=domain))

        # Output results
        if json_output:
            _output_json(xml, state, output)
        else:
            _output_text(xml, state, output)

    except Exception as e:
        logger.exception("Agent processing failed")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("bpmn_file", type=click.Path(exists=True))
@click.option(
    "--format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
def validate(bpmn_file: str, format: str) -> None:
    """
    Validate a BPMN XML file for correctness.

    \b
    Examples:
        bpmn-agent validate diagram.bpmn
        bpmn-agent validate diagram.bpmn --format json
    """
    try:
        with open(bpmn_file, "r") as f:
            xml_content = f.read()

        # Basic validation (can be extended)
        validation_result = {
            "file": bpmn_file,
            "valid": xml_content.startswith("<?xml") or "<definitions" in xml_content,
            "size_bytes": len(xml_content.encode("utf-8")),
            "element_count": xml_content.count("<"),
        }

        if format == "json":
            click.echo(json.dumps(validation_result, indent=2))
        else:
            click.echo(f"File: {validation_result['file']}")
            click.echo(f"Valid: {validation_result['valid']}")
            click.echo(f"Size: {validation_result['size_bytes']} bytes")
            click.echo(f"Elements: {validation_result['element_count']}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--llm-provider",
    type=click.Choice(["ollama", "openai_compatible"]),
    default="ollama",
    help="LLM provider",
)
@click.option(
    "--llm-base-url",
    default=None,
    help="LLM base URL",
)
@click.option(
    "--llm-model",
    default="mistral",
    help="LLM model name",
)
def health(llm_provider: str, llm_base_url: Optional[str], llm_model: str) -> None:
    """Check agent health status and LLM connectivity."""

    llm_config = LLMConfig(
        provider=llm_provider,
        base_url=llm_base_url or _get_default_llm_url(llm_provider),
        model=llm_model,
    )

    agent_config = AgentConfig(llm_config=llm_config)
    agent = BPMNAgent(agent_config)

    try:
        health_status = asyncio.run(agent.health_check())
        click.echo(json.dumps(health_status, indent=2))
    except Exception as e:
        click.echo(f"Agent health check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def info() -> None:
    """Show agent version and configuration information."""
    from bpmn_agent import __version__

    info_dict = {
        "name": "BPMN Agent",
        "version": __version__,
        "description": "Transform natural language to BPMN 2.0 diagrams",
        "modes": ["standard", "kb_enhanced", "analysis_only", "validation_only"],
        "llm_providers": ["ollama", "openai_compatible"],
        "features": {
            "knowledge_base": True,
            "domain_classification": True,
            "xml_generation": True,
            "diagram_interchange": True,
        },
    }

    click.echo(json.dumps(info_dict, indent=2))


# ==================
# Helper Functions
# ==================


def _get_default_llm_url(provider: str) -> str:
    """Get default LLM URL for provider."""
    if provider == "ollama":
        return "http://localhost:11434"
    elif provider == "openai_compatible":
        return "https://api.openai.com/v1"
    else:
        return ""


def _output_text(
    xml: Optional[str],
    state: "AgentState",
    output_file: Optional[str],
) -> None:
    """Output results as plain text."""
    if xml:
        if output_file:
            with open(output_file, "w") as f:
                f.write(xml)
            click.echo(f"BPMN XML written to: {output_file}")
        else:
            click.echo(xml)

    # Show state summary
    summary = state.summary()
    click.echo("\n--- Processing Summary ---", err=True)
    click.echo(f"Status: {'✓ Complete' if state.is_complete else '✗ Failed'}", err=True)
    click.echo(
        f"Completion: {summary['completion_rate']*100:.0f}% ({summary['stages_completed']}/{summary['stages_total']})",
        err=True,
    )

    if summary["error_count"] > 0:
        click.echo(f"Errors: {summary['error_count']}", err=True)

    if summary["warning_count"] > 0:
        click.echo(f"Warnings: {summary['warning_count']}", err=True)


def _output_json(
    xml: Optional[str],
    state: "AgentState",
    output_file: Optional[str],
) -> None:
    """Output results as JSON."""
    summary = state.summary()

    output = {
        "status": "complete" if state.is_complete else "failed",
        "summary": summary,
        "xml": xml,
    }

    output_json = json.dumps(output, indent=2)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_json)
        click.echo(f"JSON output written to: {output_file}")
    else:
        click.echo(output_json)


if __name__ == "__main__":
    cli()
