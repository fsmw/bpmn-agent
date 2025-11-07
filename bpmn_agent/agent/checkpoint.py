"""
Checkpoint and Persistence Support

Enables saving and restoring pipeline state for long-running workflows,
debugging, and resumption from specific stages.
"""

import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointType(str, Enum):
    """Types of checkpoints to save."""

    FULL = "full"  # Save complete state
    STAGE_OUTPUT = "stage_output"  # Save only stage outputs
    METRICS_ONLY = "metrics_only"  # Save only metrics


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""

    checkpoint_id: str
    timestamp: datetime
    stage_name: str
    stage_index: int
    total_stages: int
    input_hash: str
    checkpoint_type: CheckpointType
    duration_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A saved checkpoint of pipeline state."""

    metadata: CheckpointMetadata
    stage_result: Any
    state_dict: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages saving and loading checkpoints for pipeline resumption.

    Supports:
    - Saving intermediate stage results
    - Resuming from specific checkpoints
    - Garbage collection of old checkpoints
    - State serialization and deserialization
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 10,
        auto_cleanup: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum checkpoints to keep per session
            auto_cleanup: Automatically clean old checkpoints
        """
        if checkpoint_dir is None:
            checkpoint_dir = Path.home() / ".cache" / "bpmn-agent" / "checkpoints"

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.auto_cleanup = auto_cleanup
        self.checkpoint_index: Dict[str, List[CheckpointMetadata]] = {}

        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")

    def save_checkpoint(
        self,
        session_id: str,
        stage_name: str,
        stage_index: int,
        total_stages: int,
        stage_result: Any,
        state_dict: Dict[str, Any],
        input_hash: str,
        duration_ms: float,
        checkpoint_type: CheckpointType = CheckpointType.FULL,
    ) -> str:
        """Save a checkpoint of the pipeline state.

        Args:
            session_id: Unique session identifier
            stage_name: Name of the completed stage
            stage_index: Index of the stage (0-4 for 5-stage pipeline)
            total_stages: Total number of stages
            stage_result: Output from the stage
            state_dict: Dictionary of state to save
            input_hash: Hash of input for comparison
            duration_ms: Time taken by stage
            checkpoint_type: Type of checkpoint to save

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{session_id}_{stage_index}_{datetime.now().timestamp()}"

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            stage_name=stage_name,
            stage_index=stage_index,
            total_stages=total_stages,
            input_hash=input_hash,
            checkpoint_type=checkpoint_type,
            duration_ms=duration_ms,
        )

        checkpoint = Checkpoint(
            metadata=metadata,
            stage_result=stage_result,
            state_dict=state_dict,
        )

        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(session_id, checkpoint_id)

        try:
            # Serialize checkpoint
            serialized = self._serialize_checkpoint(checkpoint)
            checkpoint_path.write_bytes(serialized)

            logger.info(f"Saved checkpoint {checkpoint_id} for stage {stage_name}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            return ""

        # Update index
        if session_id not in self.checkpoint_index:
            self.checkpoint_index[session_id] = []

        self.checkpoint_index[session_id].append(metadata)

        # Cleanup if needed
        if self.auto_cleanup:
            self._cleanup_old_checkpoints(session_id)

        return checkpoint_id

    def load_checkpoint(self, session_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from disk.

        Args:
            session_id: Session identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            Loaded checkpoint or None if not found
        """
        checkpoint_path = self._get_checkpoint_path(session_id, checkpoint_id)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_id} not found")
            return None

        try:
            serialized = checkpoint_path.read_bytes()
            checkpoint = self._deserialize_checkpoint(serialized)
            logger.info(f"Loaded checkpoint {checkpoint_id}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def list_checkpoints(self, session_id: str) -> List[CheckpointMetadata]:
        """List all checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of checkpoint metadata, sorted by timestamp (newest first)
        """
        if session_id not in self.checkpoint_index:
            return []

        checkpoints = self.checkpoint_index[session_id]
        return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)

    def get_latest_checkpoint(self, session_id: str) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint for a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest checkpoint metadata or None
        """
        checkpoints = self.list_checkpoints(session_id)
        return checkpoints[0] if checkpoints else None

    def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            session_id: Session identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted, False if not found
        """
        checkpoint_path = self._get_checkpoint_path(session_id, checkpoint_id)

        if not checkpoint_path.exists():
            return False

        try:
            checkpoint_path.unlink()

            # Remove from index
            if session_id in self.checkpoint_index:
                self.checkpoint_index[session_id] = [
                    m for m in self.checkpoint_index[session_id] if m.checkpoint_id != checkpoint_id
                ]

            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def cleanup_session(self, session_id: str) -> int:
        """Clean up all checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(session_id)
        deleted_count = 0

        for metadata in checkpoints:
            if self.delete_checkpoint(session_id, metadata.checkpoint_id):
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} checkpoints for session {session_id}")
        return deleted_count

    def get_checkpoint_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about checkpoints for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with checkpoint statistics
        """
        checkpoints = self.list_checkpoints(session_id)

        if not checkpoints:
            return {
                "total_checkpoints": 0,
                "stages_saved": [],
            }

        stages_saved = list({m.stage_name for m in checkpoints})
        total_duration = sum(m.duration_ms for m in checkpoints)

        return {
            "total_checkpoints": len(checkpoints),
            "stages_saved": stages_saved,
            "latest_checkpoint": checkpoints[0].checkpoint_id if checkpoints else None,
            "total_duration_ms": total_duration,
            "oldest_checkpoint": checkpoints[-1].timestamp.isoformat(),
            "newest_checkpoint": checkpoints[0].timestamp.isoformat(),
        }

    def _get_checkpoint_path(self, session_id: str, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint.

        Args:
            session_id: Session identifier
            checkpoint_id: Checkpoint identifier

        Returns:
            Path to checkpoint file
        """
        session_dir = self.checkpoint_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"{checkpoint_id}.pkl"

    def _cleanup_old_checkpoints(self, session_id: str) -> None:
        """Remove old checkpoints exceeding max limit.

        Args:
            session_id: Session identifier
        """
        checkpoints = self.list_checkpoints(session_id)

        if len(checkpoints) > self.max_checkpoints:
            # Keep newest, delete oldest
            to_delete = checkpoints[self.max_checkpoints :]

            for metadata in to_delete:
                self.delete_checkpoint(session_id, metadata.checkpoint_id)

            logger.info(f"Cleaned up {len(to_delete)} old checkpoints for session {session_id}")

    def _serialize_checkpoint(self, checkpoint: Checkpoint) -> bytes:
        """Serialize checkpoint to bytes.

        Args:
            checkpoint: Checkpoint to serialize

        Returns:
            Serialized bytes
        """
        # Use pickle for complex objects
        return pickle.dumps(checkpoint)

    def _deserialize_checkpoint(self, data: bytes) -> Checkpoint:
        """Deserialize checkpoint from bytes.

        Args:
            data: Serialized checkpoint data

        Returns:
            Deserialized checkpoint
        """
        return pickle.loads(data)


class ResumableAgent:
    """Wrapper that adds resumption capabilities to an agent."""

    def __init__(self, agent: Any, checkpoint_manager: CheckpointManager):
        """Initialize resumable agent.

        Args:
            agent: The underlying BPMN agent
            checkpoint_manager: Manager for checkpoints
        """
        self.agent = agent
        self.checkpoint_manager = checkpoint_manager

    async def resume_from_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> tuple[Optional[str], Any]:
        """Resume processing from a checkpoint.

        Args:
            session_id: Session to resume
            checkpoint_id: Specific checkpoint to resume from (latest if None)

        Returns:
            (xml_output, state) tuple
        """
        # Get checkpoint to resume from
        if checkpoint_id is None:
            metadata = self.checkpoint_manager.get_latest_checkpoint(session_id)
            if metadata is None:
                logger.error(f"No checkpoints found for session {session_id}")
                return None, None
            checkpoint_id = metadata.checkpoint_id

        checkpoint = self.checkpoint_manager.load_checkpoint(session_id, checkpoint_id)
        if checkpoint is None:
            logger.error(f"Failed to load checkpoint {checkpoint_id}")
            return None, None

        logger.info(
            f"Resuming from checkpoint {checkpoint_id} "
            f"(completed {checkpoint.metadata.stage_index + 1}/{checkpoint.metadata.total_stages} stages)"
        )

        # Resume from next stage after checkpoint
        # This would need to be integrated into the orchestrator
        # For now, just return the checkpoint data
        return checkpoint.stage_result, checkpoint.state_dict
