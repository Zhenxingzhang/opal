"""SessionLogger — centralised file I/O for session logging.

Owns all path construction and file writing for:
- Per-call LLM logs  (``llm_calls/llm_call_{n}.json``)
- Full trajectory     (``{session_id}_trajectory.json``)
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from opal.session.session import SessionState

logger = logging.getLogger(__name__)


class SessionLogger:
    """Writes all accumulated session data to disk."""

    def __init__(self, output_dir: Path, session_id: str) -> None:
        self._session_id = session_id
        self._session_dir = output_dir / session_id[:8]

    def flush(self, session_state: SessionState) -> None:
        """Write all LLM call records and the trajectory to disk."""
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._write_llm_calls(session_state)
        self._write_trajectory(session_state)

    def _write_llm_calls(self, session_state: SessionState) -> None:
        """Write each LLM call record to ``llm_calls/llm_call_{n}.json``."""
        if not session_state.llm_calls:
            return

        llm_calls_dir = self._session_dir / "llm_calls"
        llm_calls_dir.mkdir(parents=True, exist_ok=True)

        for metrics in session_state.llm_calls:
            output_file = llm_calls_dir / f"llm_call_{metrics.call_number}.json"
            with open(output_file, "w") as f:
                json.dump(asdict(metrics), f, indent=2)

        logger.info(
            f"Wrote {len(session_state.llm_calls)} LLM call logs to {llm_calls_dir}"
        )

    def _write_trajectory(self, session_state: SessionState) -> None:
        """Write the full trajectory to ``{id}_trajectory.json``."""
        trajectory_data = {
            "session_id": session_state.id,
            "metadata": session_state.metadata,
            "trajectory": session_state.get_trajectory_as_dicts(),
        }

        output_file = self._session_dir / f"{self._session_id[:8]}_trajectory.json"
        with open(output_file, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        logger.info(f"Trajectory logged to {output_file}")
