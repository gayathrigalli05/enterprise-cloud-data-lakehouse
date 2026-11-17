
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

ROOT = Path(__file__).resolve().parents[2]

CDC_SOURCE = ROOT / "data/sample/orders_cdc.csv"
CDC_LAKEHOUSE = ROOT / "data/output/cdc_lakehouse"


def calculate_file_hash(file_path):
    """Generate a stable SHA-256 hash for a source file."""
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
        os.environ.get("CDC_SOURCE", str(CDC_SOURCE))
    )

    lakehouse_path = Path(
        os.environ.get(
            "CDC_LAKEHOUSE", str(CDC_LAKEHOUSE)
        )
    )

    if not source_path.is_file():
        raise FileNotFoundError(
            f"CDC source file not found: {source_path}"
        )

    batch_id = os.environ.get(
        "CDC_BATCH_ID", source_path.name
    )

    if not batch_id.strip():
        raise ValueError("CDC_BATCH_ID cannot be empty")

    source_hash = calculate_file_hash(source_path)

    spark = create_spark()

    try:
        if completed_batch(
            spark, lakehouse_path, batch_id, source_hash
        ):
            print(
                f"Batch {batch_id} already completed. "
                "Skipping duplicate processing."
            )
            return

        print(f"Processing CDC batch: {batch_id}")

        events = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(source_path))
        )

        # Do not checkpoint an empty source.
        if events.limit(1).count() == 0:
            raise ValueError("CDC source batch is empty")

        silver, gold = apply_cdc(
            spark, events, lakehouse_path
        )

        # Force evaluation before checkpointing.
        active_count = silver.filter(
            ~F.col("is_deleted")
        ).count()

        gold_count = gold.count()

        print("Current Silver order states:")
        silver.orderBy("order_id").show(
            truncate=False
        )

        print(f"Active order count: {active_count}")
        print(f"Gold product count: {gold_count}")

        print("Gold product revenue:")
        gold.orderBy("product").show(
            truncate=False
        )

        # Record completion only after CDC succeeds.
        mark_completed(
            spark, lakehouse_path, batch_id, source_hash
        )

        print(
            f"CDC batch {batch_id} completed "
            "and checkpoint saved successfully."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
