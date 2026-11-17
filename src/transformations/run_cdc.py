
import hashlib
import os
from pathlib import Path

from pyspark.sql import functions as F

from src.transformations.delta_pipeline import create_spark
from src.transformations.cdc_pipeline import apply_cdc
from src.transformations.cdc_checkpoint import (
    completed_batch,
    mark_completed,
)
from src.monitoring.pipeline_monitor import PipelineMonitor


ROOT = Path(__file__).resolve().parents[2]

CDC_SOURCE = ROOT / "data/sample/orders_cdc.csv"
CDC_LAKEHOUSE = ROOT / "data/output/cdc_lakehouse"


def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of the CDC source."""
    digest = hashlib.sha256()

    with open(file_path, "rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    source_path = Path(
        os.environ.get(
            "CDC_SOURCE",
            str(CDC_SOURCE),
        )
    )

    lakehouse_path = Path(
        os.environ.get(
            "CDC_LAKEHOUSE",
            str(CDC_LAKEHOUSE),
        )
    )

    batch_id = os.environ.get(
        "CDC_BATCH_ID",
        source_path.name,
    )

    monitor = PipelineMonitor(
        pipeline_name="CDC Orders Pipeline",
        batch_id=batch_id,
        log_directory=(
            lakehouse_path / "monitoring" / "audit_logs"
        ),
    )

    monitor.start()
    spark = None

    try:
        if not batch_id.strip():
            raise ValueError(
                "CDC_BATCH_ID cannot be empty"
            )

        if not source_path.is_file():
            raise FileNotFoundError(
                f"CDC source file not found: {source_path}"
            )

        source_hash = calculate_file_hash(
            source_path
        )

        monitor.record_metric(
            "source_file",
            source_path.name,
        )

        monitor.record_metric(
            "source_hash",
            source_hash,
        )

        spark = create_spark()

        # Check whether the batch was completed.
        if completed_batch(
            spark,
            lakehouse_path,
            batch_id,
            source_hash,
        ):
            monitor.record_metric(
                "checkpoint_status",
                "ALREADY_COMPLETED",
            )

            monitor.record_metric(
                "processing_skipped",
                True,
            )

            print(
                f"Batch {batch_id} already completed. "
                "Skipping duplicate processing."
            )

            monitor.success()
            return

        monitor.record_metric(
            "checkpoint_status",
            "NOT_COMPLETED",
        )

        monitor.record_metric(
            "processing_skipped",
            False,
        )

        print(
            f"Starting CDC batch: {batch_id}"
        )

        events = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(source_path))
        )

        incoming_count = events.count()

        monitor.record_metric(
            "incoming_event_count",
            incoming_count,
        )

        if incoming_count == 0:
            raise ValueError(
                "CDC source batch is empty"
            )

        # Process Bronze, Silver and Gold.
        silver, gold = apply_cdc(
            spark,
            events,
            lakehouse_path,
        )

        bronze = (
            spark.read.format("delta")
            .load(
                str(
                    lakehouse_path / "bronze/events"
                )
            )
        )

        bronze_count = bronze.count()
        silver_count = silver.count()
        gold_count = gold.count()

        active_count = silver.filter(
            ~F.col("is_deleted")
        ).count()

        deleted_count = silver.filter(
            F.col("is_deleted")
        ).count()

        total_revenue = (
            gold.agg(
                F.sum("total_revenue").alias(
                    "revenue"
                )
            )
            .collect()[0]["revenue"]
        )

        monitor.record_metric(
            "bronze_event_count",
            bronze_count,
        )

        monitor.record_metric(
            "silver_order_count",
            silver_count,
        )

        monitor.record_metric(
            "gold_product_count",
            gold_count,
        )

        monitor.record_metric(
            "active_order_count",
            active_count,
        )

        monitor.record_metric(
            "deleted_order_count",
            deleted_count,
        )

        monitor.record_metric(
            "total_revenue",
            float(total_revenue or 0.0),
        )

        print("Current Silver order states:")

        silver.orderBy("order_id").show(
            truncate=False
        )

        print("Gold product revenue:")

        gold.orderBy("product").show(
            truncate=False
        )

        # Persist checkpoint after successful processing.
        mark_completed(
            spark,
            lakehouse_path,
            batch_id,
            source_hash,
        )

        monitor.record_metric(
            "checkpoint_status",
            "COMPLETED",
        )

        # Mark the monitored execution successful.
        monitor.success()

        print(
            "CDC pipeline completed successfully."
        )

        print(
            f"Bronze events: {bronze_count}"
        )

        print(
            f"Silver orders: {silver_count}"
        )

        print(
            f"Gold products: {gold_count}"
        )

        print(
            f"Active orders: {active_count}"
        )

        print(
            f"Total revenue: {total_revenue or 0.0}"
        )

    except Exception as error:
        monitor.failure(error)

        print(
            f"CDC pipeline failed: {error}"
        )

        raise

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
