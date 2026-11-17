
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.window import Window


COLUMNS = [
    "event_id", "order_id", "customer_id",
    "product", "quantity", "unit_price",
    "operation", "sequence_num", "updated_at"
]


def prepare_events(df):
    """Validate and normalize incoming CDC events."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = (
        df.select(*COLUMNS)
        .withColumn("order_id", F.col("order_id").cast("long"))
        .withColumn("quantity", F.col("quantity").cast("int"))
        .withColumn("unit_price", F.col("unit_price").cast("double"))
        .withColumn("sequence_num", F.col("sequence_num").cast("long"))
        .withColumn("updated_at", F.to_timestamp("updated_at"))
    )

    invalid = (
        F.col("event_id").isNull()
        | F.col("order_id").isNull()
        | F.col("sequence_num").isNull()
        | F.col("updated_at").isNull()
        | ~F.col("operation").isin("I", "U", "D")
        | (
            (F.col("operation") != "D")
            & (
                F.col("product").isNull()
                | F.col("quantity").isNull()
                | F.col("unit_price").isNull()
                | (F.col("quantity") <= 0)
                | (F.col("unit_price") < 0)
            )
        )
    )

    if df.filter(invalid).limit(1).count():
        raise ValueError("Invalid CDC event")

    # Remove exact duplicates only.
    df = df.dropDuplicates()

    # Each event ID must describe one unique event.
    if (
        df.groupBy("event_id")
        .count()
        .filter(F.col("count") > 1)
        .limit(1).count()
    ):
        raise ValueError("Conflicting event_id values")

    # Sequence numbers must be unique per order.
    if (
        df.groupBy("order_id", "sequence_num")
        .count()
        .filter(F.col("count") > 1)
        .limit(1).count()
    ):
        raise ValueError("Conflicting order sequence numbers")

    return df


def latest_events(events):
    """Keep the highest-sequence event for each order."""
    window = (
        Window.partitionBy("order_id")
        .orderBy(F.col("sequence_num").desc())
    )

    return (
        events.withColumn(
            "_rank", F.row_number().over(window)
        )
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def apply_cdc(spark, incoming, base_path):
    """Apply CDC events to Bronze, Silver and Gold."""
    base = Path(base_path)
    bronze_path = str(base / "bronze/events")
    silver_path = str(base / "silver/orders")
    gold_path = str(base / "gold/product_revenue")

    incoming = prepare_events(incoming)

    # Bronze: store each event once.
    if DeltaTable.isDeltaTable(spark, bronze_path):
        bronze = DeltaTable.forPath(spark, bronze_path)

        (
            bronze.alias("t")
            .merge(
                incoming.alias("s"),
                "t.event_id = s.event_id"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        (
            incoming.write.format("delta")
            .mode("errorifexists")
            .save(bronze_path)
        )

    # Read the full event history. This also makes
    # repeated runs and out-of-order delivery safe
    # within this single-writer demonstration.
    history = (
        spark.read.format("delta")
        .load(bronze_path)
    )

    # Reject conflicting events across batches.
    if (
        history.groupBy("order_id", "sequence_num")
        .count()
        .filter(F.col("count") > 1)
        .limit(1).count()
    ):
        raise ValueError("Conflicting CDC event history")

    latest = latest_events(history)

    silver = (
        latest.select(
            "order_id", "customer_id", "product",
            "quantity", "unit_price", "operation",
            "sequence_num", "updated_at"
        )
        .withColumn(
            "is_deleted",
            F.col("operation") == "D"
        )
        .withColumn(
            "revenue",
            F.when(
                F.col("operation") != "D",
                F.col("quantity") * F.col("unit_price")
            ).otherwise(F.lit(0.0))
        )
    )

    # Silver: upsert the latest order state.
    if DeltaTable.isDeltaTable(spark, silver_path):
        target = DeltaTable.forPath(spark, silver_path)

        (
            target.alias("t")
            .merge(
                silver.alias("s"),
                "t.order_id = s.order_id"
            )
            .whenMatchedUpdateAll(
                condition=(
                    "s.sequence_num > t.sequence_num"
                )
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        (
            silver.write.format("delta")
            .mode("errorifexists")
            .save(silver_path)
        )

    current = (
        spark.read.format("delta")
        .load(silver_path)
    )

    # Gold: exclude deleted orders.
    gold = (
        current.filter(~F.col("is_deleted"))
        .groupBy("product")
        .agg(
            F.sum("revenue").alias("total_revenue")
        )
    )

    # Full refresh avoids stale product totals.
    (
        gold.write.format("delta")
        .mode("overwrite")
        .save(gold_path)
    )

    return current, gold
