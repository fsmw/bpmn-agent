"""Pytest configuration for bpmn-agent tests."""

import sys
from pathlib import Path
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Add the bpmn-agent source directory to the path
bpmn_agent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(bpmn_agent_dir))

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


def pytest_configure(config):
    """Configure pytest with asyncio support."""
    # Set asyncio mode to auto for pytest-asyncio
    config.option.asyncio_mode = "auto"


# ===========================
# Fixtures for E2E tests with mocked LLM
# ===========================


@pytest.fixture
def mock_llm_extraction_response():
    """Create a mock LLM extraction response with realistic entities."""

    def _create_response(process_type="generic"):
        """Generate response JSON based on process type."""
        if process_type == "approval":
            return json.dumps(
                {
                    "entities": [
                        {
                            "id": "actor_1",
                            "name": "Employee",
                            "type": "actor",
                            "confidence": "high",
                        },
                        {"id": "actor_2", "name": "Manager", "type": "actor", "confidence": "high"},
                        {"id": "actor_3", "name": "HR", "type": "actor", "confidence": "high"},
                        {
                            "id": "task_1",
                            "name": "submit request",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_2",
                            "name": "review",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_3",
                            "name": "process",
                            "type": "activity",
                            "confidence": "medium",
                        },
                        {
                            "id": "decision_1",
                            "name": "decision",
                            "type": "gateway",
                            "confidence": "medium",
                        },
                    ],
                    "relations": [
                        {
                            "id": "rel_1",
                            "source_id": "actor_1",
                            "target_id": "task_1",
                            "type": "involves",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_2",
                            "source_id": "actor_2",
                            "target_id": "task_2",
                            "type": "involves",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_3",
                            "source_id": "task_1",
                            "target_id": "decision_1",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_4",
                            "source_id": "decision_1",
                            "target_id": "task_3",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                    ],
                }
            )
        elif process_type == "parallel":
            return json.dumps(
                {
                    "entities": [
                        {"id": "actor_1", "name": "System", "type": "actor", "confidence": "high"},
                        {
                            "id": "task_1",
                            "name": "receive order",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_2",
                            "name": "validate inventory",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_3",
                            "name": "check payment",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_4",
                            "name": "pick items",
                            "type": "activity",
                            "confidence": "medium",
                        },
                        {
                            "id": "task_5",
                            "name": "ship",
                            "type": "activity",
                            "confidence": "medium",
                        },
                        {
                            "id": "gateway_1",
                            "name": "parallel fork",
                            "type": "gateway",
                            "confidence": "medium",
                        },
                        {
                            "id": "gateway_2",
                            "name": "parallel join",
                            "type": "gateway",
                            "confidence": "medium",
                        },
                    ],
                    "relations": [
                        {
                            "id": "rel_1",
                            "source_id": "actor_1",
                            "target_id": "task_1",
                            "type": "involves",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_2",
                            "source_id": "task_1",
                            "target_id": "gateway_1",
                            "type": "precedes",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_3",
                            "source_id": "gateway_1",
                            "target_id": "task_2",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_4",
                            "source_id": "gateway_1",
                            "target_id": "task_3",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_5",
                            "source_id": "task_2",
                            "target_id": "gateway_2",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_6",
                            "source_id": "task_3",
                            "target_id": "gateway_2",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_7",
                            "source_id": "gateway_2",
                            "target_id": "task_4",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_8",
                            "source_id": "task_4",
                            "target_id": "task_5",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                    ],
                }
            )
        else:  # generic
            return json.dumps(
                {
                    "entities": [
                        {
                            "id": "actor_1",
                            "name": "Customer",
                            "type": "actor",
                            "confidence": "high",
                        },
                        {"id": "actor_2", "name": "System", "type": "actor", "confidence": "high"},
                        {
                            "id": "task_1",
                            "name": "submit order",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_2",
                            "name": "process payment",
                            "type": "activity",
                            "confidence": "high",
                        },
                        {
                            "id": "task_3",
                            "name": "ship order",
                            "type": "activity",
                            "confidence": "medium",
                        },
                    ],
                    "relations": [
                        {
                            "id": "rel_1",
                            "source_id": "actor_1",
                            "target_id": "task_1",
                            "type": "involves",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_2",
                            "source_id": "actor_2",
                            "target_id": "task_2",
                            "type": "involves",
                            "confidence": "high",
                        },
                        {
                            "id": "rel_3",
                            "source_id": "task_1",
                            "target_id": "task_2",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                        {
                            "id": "rel_4",
                            "source_id": "task_2",
                            "target_id": "task_3",
                            "type": "precedes",
                            "confidence": "medium",
                        },
                    ],
                }
            )

    return _create_response


@pytest.fixture
def mock_llm_client(mock_llm_extraction_response):
    """Create a mock LLM client that returns valid extraction responses."""
    mock_client = AsyncMock()

    # Set up config attribute
    mock_config = MagicMock()
    mock_config.model = "mistral"
    mock_config.provider = "ollama"
    mock_client.config = mock_config

    # Track call count to help debug
    call_count = [0]  # Using list to allow modification in nested function

    # Default behavior: return a generic response
    async def mock_call(messages, temperature=0.3, max_tokens=4096, **kwargs):
        call_count[0] += 1
        # Analyze messages to determine process type
        process_type = "generic"
        content_str = ""

        if messages:
            # Get the user message (should be messages[-1])
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                content_str = last_msg.content
            elif isinstance(last_msg, dict) and "content" in last_msg:
                content_str = last_msg["content"]
            else:
                content_str = str(last_msg)

            # Extract only the actual process description from the full message
            # The message format is: [instructions] + "Process Description:" + [description] + "Return ONLY..."
            process_description = ""
            if "Process Description:" in content_str:
                parts = content_str.split("Process Description:")
                if len(parts) > 1:
                    # Get the description, stopping at "Return ONLY" or similar markers
                    description_part = parts[1]
                    for marker in ["Return ONLY", "return ONLY", "ONLY return"]:
                        if marker in description_part:
                            description_part = description_part.split(marker)[0]
                            break
                    process_description = description_part.strip()
            else:
                # Fallback: use entire message if we can't find the marker
                process_description = content_str

            # Debug logging
            import sys

            desc_preview = (
                process_description[:200] if len(process_description) > 200 else process_description
            )
            print(
                f"\n[MOCK DEBUG {call_count[0]}] Process description: {desc_preview}...",
                file=sys.stderr,
            )

            content = process_description.lower()

            # Check for approval workflow
            if (
                "approval" in content
                or "approve" in content
                or "reject" in content
                or "manager" in content
            ):
                process_type = "approval"
            # Check for parallel workflow
            elif (
                "parallel" in content
                or "inventory" in content
                or ("payment" in content and "inventory" in content)
            ):
                process_type = "parallel"

            print(
                f"[MOCK DEBUG {call_count[0]}] Detected process_type={process_type}\n",
                file=sys.stderr,
            )

        response_json = mock_llm_extraction_response(process_type)

        # Create mock response object
        mock_response = MagicMock()
        mock_response.content = response_json
        return mock_response

    mock_client.call = mock_call
    return mock_client


@pytest.fixture
def patch_llm_for_e2e(mock_llm_client):
    """Patch the LLM client factory to use mock client."""
    with patch("bpmn_agent.core.llm_client.LLMClientFactory.create", return_value=mock_llm_client):
        yield mock_llm_client
