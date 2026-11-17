
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def prepare_updates(df):
    """Validate and deterministically deduplicate updates."""
    required = {
        "order_id", "customer_id", "product",
        "quantity", "unit_price", "updated_at"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.withColumn(
        "updated_at", F.to_timestamp("updated_at")
    )

    if df.filter(
        F.col("order_id").isNull()
        | F.col("updated_at").isNull()
    ).limit(1).count():
        raise ValueError("Invalid order_id or updated_at")

    # Identical duplicates are safe to collapse.
    df = df.dropDuplicates()

    # Conflicting records with the same order_id and
    # updated_at cannot be ordered reliably.
    conflicts = (
        df.groupBy("order_id", "updated_at")
        .count()
        .filter(F.col("count") > 1)
    )
    if conflicts.limit(1).count():
        raise ValueError(
            "Conflicting updates share the same timestamp"
        )

    window = Window.partitionBy("order_id").orderBy(
        F.col("updated_at").desc()
    )

    return (
        df.withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def merge_incremental(spark, updates, target_path):
    """Insert new orders and apply strictly newer updates."""
    target_path = str(Path(target_path))
    updates = prepare_updates(updates)

    if DeltaTable.isDeltaTable(spark, target_path):
        target = DeltaTable.forPath(spark, target_path)

        (
            target.alias("t")
            .merge(
                updates.alias("s"),
                "t.order_id = s.order_id"
            )
            .whenMatchedUpdateAll(
                condition="s.updated_at > t.updated_at"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        (
            updates.write.format("delta")
            .mode("errorifexists")
            .save(target_path)
        )


def get_watermark(spark, target_path):
    """Return the latest processed timestamp."""
    target_path = str(Path(target_path))

    if not DeltaTable.isDeltaTable(spark, target_path):
        return None

    return (
        spark.read.format("delta")
        .load(target_path)
        .agg(F.max("updated_at").alias("watermark"))
        .first()["watermark"]
    )


def filter_new_records(df, watermark):
    """Strict watermark filter for ordered input batches."""
    df = df.withColumn(
        "updated_at", F.to_timestamp("updated_at")
    )
    if watermark is None:
        return df
    return df.filter(F.col("updated_at") > F.lit(watermark))
