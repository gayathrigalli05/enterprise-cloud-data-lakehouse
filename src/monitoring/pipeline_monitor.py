
import json
import logging
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path


logger = logging.getLogger(__name__)


def utc_now():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class PipelineMonitor:
    """
    Track pipeline execution metrics and persist
    structured JSON audit logs.

    Designed for local, single-process execution.
    """

    def __init__(
        self,
        pipeline_name,
        batch_id,
        log_directory="data/output/monitoring",
    ):
        self.pipeline_name = pipeline_name
        self.batch_id = batch_id
        self.execution_id = str(uuid.uuid4())

        self.log_directory = Path(log_directory)

        self.status = "PENDING"
        self.start_time = None
        self.end_time = None
        self.duration_seconds = None

        self.metrics = {}
        self.error_message = None

        self._start_clock = None

    def start(self):
        """Mark pipeline execution as running."""
        if self.status != "PENDING":
            raise RuntimeError(
                "Pipeline execution has already started"
            )

        self.status = "RUNNING"
        self.start_time = utc_now()
        self._start_clock = time.monotonic()

        logger.info(
            "Pipeline started: %s, execution=%s",
            self.pipeline_name,
            self.execution_id,
        )

        return self

    def record_metric(self, metric_name, value):
        """Record a JSON-serializable pipeline metric."""
        if self.status != "RUNNING":
            raise RuntimeError(
                "Metrics can only be recorded while running"
            )

        if not metric_name:
            raise ValueError(
                "Metric name cannot be empty"
            )

        # Fail early if the metric cannot be serialized.
        json.dumps(value, allow_nan=False)

        self.metrics[metric_name] = value

    def _finish(self, status, error_message=None):
        """Finalize an execution and persist its audit log."""
        if self.status != "RUNNING":
            raise RuntimeError(
                "Pipeline must be running before completion"
            )

        self.end_time = utc_now()

        self.duration_seconds = round(
            time.monotonic() - self._start_clock,
            3,
        )

        self.status = status
        self.error_message = error_message

        self.save_audit_log()

        logger.info(
            "Pipeline finished: %s, status=%s",
            self.pipeline_name,
            self.status,
        )

    def success(self):
        """Mark execution as successful."""
        self._finish("SUCCESS")

    def failure(self, error):
        """Mark execution as failed."""
        self._finish(
            "FAILED",
            error_message=str(error),
        )

    def to_dict(self):
        """Return a structured audit record."""
        return {
            "execution_id": self.execution_id,
            "pipeline_name": self.pipeline_name,
            "batch_id": self.batch_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "error_message": self.error_message,
        }

    def save_audit_log(self):
        """
        Persist an execution record to a unique JSON file.

        Uses atomic replacement on the local filesystem
        to avoid exposing partially written JSON.
        """
        self.log_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_path = (
            self.log_directory
            / f"{self.execution_id}.json"
        )

        temporary_path = (
            self.log_directory
            / f"{self.execution_id}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.to_dict(),
                    file,
                    indent=2,
                    allow_nan=False,
                )

            temporary_path.replace(final_path)

        finally:
            temporary_path.unlink(missing_ok=True)

        return final_path
