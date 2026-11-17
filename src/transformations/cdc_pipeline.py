
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.window import Window


COLUMNS = [
    "event_id",
    "order_id",
    "customer_id",
    "product",
    "quantity",
    "unit_price",
    "operation",
    "sequence_num",
    "updated_at",
]


def inject_failure(fail_after, stage):
    """Simulate a failure after a completed pipeline stage."""
    if fail_after == stage:
        raise RuntimeError(
            f"Simulated CDC failure after {stage}"
        )


def prepare_events(df):
    """Validate and normalize incoming CDC events."""

    missing = set(COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

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
        | F.col("operation").isNull()
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

    df = df.dropDuplicates()

    event_conflicts = (
        df.groupBy("event_id")
        .count()
        .filter(F.col("count") > 1)
    )

    if event_conflicts.limit(1).count():
        raise ValueError(
            "Conflicting event_id values"
        )

    sequence_conflicts = (
        df.groupBy("order_id", "sequence_num")
        .count()
        .filter(F.col("count") > 1)
    )

    if sequence_conflicts.limit(1).count():
        raise ValueError(
            "Conflicting order sequence numbers"
        )

    return df


def latest_events(events):
    """Select the latest event for each order."""

    window = (
        Window.partitionBy("order_id")
        .orderBy(F.col("sequence_num").desc())
    )

    return (
        events.withColumn(
            "_rank",
            F.row_number().over(window)
        )
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def validate_event_history(spark, incoming, bronze_path):
    """Reject conflicting CDC events across batches."""

    if not DeltaTable.isDeltaTable(spark, bronze_path):
        return

    existing = (
        spark.read.format("delta")
        .load(bronze_path)
        .select(*COLUMNS)
    )

    joined_events = (
        incoming.alias("s")
        .join(
            existing.alias("t"),
            F.col("s.event_id") == F.col("t.event_id"),
            "inner"
        )
    )

    differences = [
        ~F.col(f"s.{column}").eqNullSafe(
            F.col(f"t.{column}")
        )
        for column in COLUMNS
        if column != "event_id"
    ]

    conflict_condition = differences[0]

    for difference in differences[1:]:
        conflict_condition = (
            conflict_condition | difference
        )

    if joined_events.filter(
        conflict_condition
    ).limit(1).count():
        raise ValueError(
            "Conflicting CDC event_id across batches"
        )

    sequence_conflicts = (
        incoming.alias("s")
        .join(
            existing.alias("t"),
            (
                (F.col("s.order_id") == F.col("t.order_id"))
                & (
                    F.col("s.sequence_num")
                    == F.col("t.sequence_num")
                )
            ),
            "inner"
        )
        .filter(
            F.col("s.event_id") != F.col("t.event_id")
        )
    )

    if sequence_conflicts.limit(1).count():
        raise ValueError(
            "Conflicting order sequence across batches"
        )


def apply_cdc(
    spark,
    incoming,
    base_path,
    fail_after=None
):
    """
    Apply CDC events to Bronze, Silver and Gold.

    Supports inserts, updates, soft deletes, replay,
    out-of-order events, conflict validation, and
    controlled failure injection.
    """

    if fail_after not in (None, "bronze", "silver", "gold"):
        raise ValueError(
            "fail_after must be bronze, silver, gold, or None"
        )

    base = Path(base_path)

    bronze_path = str(base / "bronze/events")
    silver_path = str(base / "silver/orders")
    gold_path = str(base / "gold/product_revenue")

    # Step 1: Validate incoming events.
    incoming = prepare_events(incoming)

    # Step 2: Validate against Bronze history.
    validate_event_history(
        spark,
        incoming,
        bronze_path
    )

    # Step 3: Write unique events to Bronze.
    if DeltaTable.isDeltaTable(spark, bronze_path):
        bronze = DeltaTable.forPath(
            spark, bronze_path
        )

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
            incoming.write
            .format("delta")
            .mode("errorifexists")
            .save(bronze_path)
        )

    # Simulate failure after Bronze commit.
    inject_failure(fail_after, "bronze")

    # Step 4: Read Bronze history.
    history = (
        spark.read.format("delta")
        .load(bronze_path)
    )

    history_conflicts = (
        history.groupBy("order_id", "sequence_num")
        .count()
        .filter(F.col("count") > 1)
    )

    if history_conflicts.limit(1).count():
        raise ValueError(
            "Conflicting CDC event history"
        )

    # Step 5: Select latest event per order.
    latest = latest_events(history)

    # Step 6: Build Silver order states.
    silver = (
        latest.select(
            "order_id",
            "customer_id",
            "product",
            "quantity",
            "unit_price",
            "operation",
            "sequence_num",
            "updated_at"
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

    # Step 7: MERGE Silver order states.
    if DeltaTable.isDeltaTable(spark, silver_path):
        target = DeltaTable.forPath(
            spark, silver_path
        )

        (
            target.alias("t")
            .merge(
                silver.alias("s"),
                "t.order_id = s.order_id"
            )
            .whenMatchedUpdateAll(
                condition="s.sequence_num > t.sequence_num"
            )
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        (
            silver.write
            .format("delta")
            .mode("errorifexists")
            .save(silver_path)
        )

    # Simulate failure after Silver commit.
    inject_failure(fail_after, "silver")

    # Step 8: Read current Silver state.
    current = (
        spark.read.format("delta")
        .load(silver_path)
    )

    # Step 9: Build Gold revenue aggregates.
    gold = (
        current.filter(~F.col("is_deleted"))
        .groupBy("product")
        .agg(
            F.sum("revenue").alias("total_revenue")
        )
    )

    # Step 10: Refresh Gold.
    (
        gold.write
        .format("delta")
        .mode("overwrite")
        .save(gold_path)
    )

    # Simulate failure after Gold commit.
    inject_failure(fail_after, "gold")

    return current, gold
