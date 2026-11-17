
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType
)

CHECKPOINT_SCHEMA = StructType([
    StructField("batch_id", StringType(), False),
    StructField("source_hash", StringType(), False),
])


def checkpoint_path(base_path):
    return str(Path(base_path) / "checkpoints" / "cdc_batches")


def completed_batch(spark, base_path, batch_id, source_hash):
    path = checkpoint_path(base_path)

    if not DeltaTable.isDeltaTable(spark, path):
        return False

    matches = (
        spark.read.format("delta").load(path)
        .filter(F.col("batch_id") == batch_id)
        .select("source_hash")
        .collect()
    )

    if not matches:
        return False

    if any(row.source_hash != source_hash for row in matches):
        raise ValueError(
            f"Batch {batch_id} was previously processed "
            "with different source content"
        )

    return True


def mark_completed(spark, base_path, batch_id, source_hash):
    path = checkpoint_path(base_path)

    data = spark.createDataFrame(
        [(batch_id, source_hash)],
        schema=CHECKPOINT_SCHEMA
    )

    if DeltaTable.isDeltaTable(spark, path):
        (
            DeltaTable.forPath(spark, path)
            .alias("t")
            .merge(
                data.alias("s"),
                "t.batch_id = s.batch_id"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        data.write.format("delta").mode(
            "errorifexists"
        ).save(path)
