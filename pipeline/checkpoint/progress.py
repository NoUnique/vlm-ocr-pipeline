"""Progress tracking for pipeline checkpoints.

This module implements ProgressTracker for managing pipeline execution state.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track pipeline execution progress with checkpoints.

    Manages _progress.json file that tracks:
    - Completed stages
    - Current/failed stage
    - Timestamps and durations
    - Error messages
    - Output file locations

    Example:
        >>> tracker = ProgressTracker(Path("results/doc1"))
        >>> tracker.start_stage("detection")
        >>> # ... run detection ...
        >>> tracker.complete_stage("detection", Path("stage2_detection.json"))
        >>>
        >>> # On resume:
        >>> resume_point = tracker.get_resume_point()
        >>> if resume_point:
        ...     stage_name, checkpoint_file = resume_point
        ...     print(f"Resuming from {stage_name}")
    """

    PROGRESS_FILE = "_progress.json"
    VERSION = "1.0"

    def __init__(self, output_dir: Path, input_file: str | None = None):
        """Initialize progress tracker.

        Args:
            output_dir: Directory for checkpoint files
            input_file: Input file path for validation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.output_dir / self.PROGRESS_FILE
        self.input_file = input_file

        # Load existing progress or initialize new
        self.data = self._load_or_init()

        # Track stage start time
        self._stage_start_time: float | None = None

    def _load_or_init(self) -> dict[str, Any]:
        """Load existing progress or initialize new progress data.

        Returns:
            Progress data dictionary
        """
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    logger.info("Loaded checkpoint from %s", self.progress_file)
                    return data
            except Exception as e:
                logger.warning("Failed to load progress file: %s", e)

        # Initialize new progress
        return {
            "version": self.VERSION,
            "input_file": self.input_file,
            "start_time": datetime.now(UTC).isoformat(),
            "last_updated": datetime.now(UTC).isoformat(),
            "status": "in_progress",
            "completed_stages": [],
            "current_stage": None,
            "failed_stage": None,
            "error": None,
            "stages": {},
        }

    def _save(self) -> None:
        """Save progress data to file."""
        self.data["last_updated"] = datetime.now(UTC).isoformat()

        try:
            with open(self.progress_file, "w") as f:
                json.dump(self.data, f, indent=2)
            logger.debug("Saved progress to %s", self.progress_file)
        except Exception as e:
            logger.error("Failed to save progress: %s", e)

    def start_stage(self, stage_name: str) -> None:
        """Mark stage as started.

        Args:
            stage_name: Name of stage starting
        """
        self.data["current_stage"] = stage_name
        self._stage_start_time = time.time()

        # Initialize stage entry
        if stage_name not in self.data["stages"]:
            self.data["stages"][stage_name] = {
                "status": "in_progress",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        self._save()
        logger.info("Started stage: %s", stage_name)

    def complete_stage(self, stage_name: str, output_file: Path | str) -> None:
        """Mark stage as completed successfully.

        Args:
            stage_name: Name of completed stage
            output_file: Path to stage output file
        """
        duration = time.time() - self._stage_start_time if self._stage_start_time else 0

        # Update stage info
        self.data["stages"][stage_name] = {
            "status": "completed",
            "duration_seconds": round(duration, 2),
            "output_file": str(output_file),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Update completed stages list
        if stage_name not in self.data["completed_stages"]:
            self.data["completed_stages"].append(stage_name)

        self.data["current_stage"] = None
        self._stage_start_time = None

        self._save()
        logger.info("Completed stage: %s (%.2fs)", stage_name, duration)

    def fail_stage(self, stage_name: str, error: Exception) -> None:
        """Mark stage as failed.

        Args:
            stage_name: Name of failed stage
            error: Exception that caused failure
        """
        duration = time.time() - self._stage_start_time if self._stage_start_time else 0

        # Update stage info
        self.data["stages"][stage_name] = {
            "status": "failed",
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now(UTC).isoformat(),
            "error": f"{type(error).__name__}: {str(error)}",
        }

        # Update overall status
        self.data["status"] = "failed"
        self.data["failed_stage"] = stage_name
        self.data["error"] = f"{type(error).__name__}: {str(error)}"
        self.data["current_stage"] = None
        self._stage_start_time = None

        self._save()
        logger.error("Failed stage: %s (%s)", stage_name, error)

    def mark_complete(self) -> None:
        """Mark entire pipeline as completed successfully."""
        self.data["status"] = "completed"
        self.data["current_stage"] = None
        self.data["failed_stage"] = None
        self.data["error"] = None

        self._save()
        logger.info("Pipeline completed successfully")

    def get_resume_point(self) -> tuple[str, Path] | None:
        """Get resume point if checkpoint exists.

        Returns:
            (stage_name, checkpoint_file) tuple, or None if no valid checkpoint

        Example:
            >>> resume_point = tracker.get_resume_point()
            >>> if resume_point:
            ...     stage, checkpoint = resume_point
            ...     print(f"Resume from {stage} using {checkpoint}")
        """
        # Check if there's a valid checkpoint
        if self.data["status"] == "completed":
            logger.info("Previous run completed successfully - starting fresh")
            return None

        if not self.data["completed_stages"]:
            logger.info("No completed stages found - starting from beginning")
            return None

        # Determine which stage to resume from
        if self.data["failed_stage"]:
            # Resume from failed stage
            resume_stage = self.data["failed_stage"]
            logger.info("Found failed stage: %s", resume_stage)
        elif self.data["current_stage"]:
            # Resume from interrupted stage
            resume_stage = self.data["current_stage"]
            logger.info("Found interrupted stage: %s", resume_stage)
        else:
            # No clear resume point
            return None

        # Find the last completed stage's output file
        completed_stages = self.data["completed_stages"]
        if not completed_stages:
            return None

        last_completed = completed_stages[-1]
        stage_info = self.data["stages"].get(last_completed)

        if not stage_info or "output_file" not in stage_info:
            logger.warning("No output file for last completed stage: %s", last_completed)
            return None

        checkpoint_file = self.output_dir / stage_info["output_file"]

        if not checkpoint_file.exists():
            logger.warning("Checkpoint file not found: %s", checkpoint_file)
            return None

        return (resume_stage, checkpoint_file)

    def validate_input(self, input_file: str) -> bool:
        """Validate that checkpoint matches input file.

        Args:
            input_file: Current input file path

        Returns:
            True if checkpoint is valid for this input file
        """
        if not self.data.get("input_file"):
            # No input file recorded - can't validate
            return True

        if self.data["input_file"] != input_file:
            logger.warning(
                "Input file mismatch: checkpoint=%s, current=%s",
                self.data["input_file"],
                input_file,
            )
            return False

        return True

    def print_resume_info(self) -> None:
        """Print human-readable resume information."""
        if self.data["status"] == "completed":
            print("\nPrevious run completed successfully")
            return

        if not self.data["completed_stages"]:
            print("\nNo checkpoint found - starting fresh")
            return

        print("\n" + "=" * 60)
        print("Found checkpoint in", self.output_dir)
        print("=" * 60)

        # Last run time
        if "start_time" in self.data:
            start_time = datetime.fromisoformat(self.data["start_time"])
            now = datetime.now(UTC)
            elapsed = now - start_time
            print(f"Last run: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Elapsed: {elapsed}")

        # Completed stages
        print(f"\nCompleted stages: {', '.join(self.data['completed_stages'])}")

        # Failed stage
        if self.data.get("failed_stage"):
            print(f"Failed at: {self.data['failed_stage']}")
            if self.data.get("error"):
                print(f"Error: {self.data['error']}")

        resume_point = self.get_resume_point()
        if resume_point:
            stage_name, _ = resume_point
            print(f"\n→ Resuming from '{stage_name}' stage...")
        print("=" * 60 + "\n")
