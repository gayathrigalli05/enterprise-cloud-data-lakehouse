
import pytest

from delta.tables import DeltaTable
from src.transformations.cdc_pipeline import apply_cdc
from src.transformations.cdc_checkpoint import (
    completed_batch,
    mark_completed,
)


SCHEMA = (
    "event_id string, "
    "order_id long, "
    "customer_id string, "
    "product string, "
    "quantity int, "
    "unit_price double, "
    "operation string, "
    "sequence_num long, "
    "updated_at string"
)


def make_events(spark, rows):
    return spark.createDataFrame(rows, schema=SCHEMA)


def event(
    event_id,
    quantity=2,
    operation="I",
    sequence=1,
):
    return (
        event_id,
        101,
        "C001",
        "Laptop",
        quantity,
        100.0,
        operation,
        sequence,
        "2026-01-01 10:00:00",
    )


@pytest.mark.parametrize(
    "stage",
    ["bronze", "silver", "gold"],
)
def test_recovery_after_stage_failure(
    spark,
    tmp_path,
    stage,
):
    """Replay must recover after each simulated failure."""
    base = str(tmp_path / stage)
    batch_id = "batch-001"
    source_hash = "test-source-hash"

    events = make_events(
        spark,
        [event("evt-001")],
    )

    with pytest.raises(
        RuntimeError,
        match=f"Simulated CDC failure after {stage}",
    ):
        apply_cdc(
            spark,
            events,
            base,
            fail_after=stage,
        )

    # A failed run must not have a completion checkpoint.
    assert not completed_batch(
        spark, base, batch_id, source_hash
    )

    # Replay the same batch without failure injection.
    silver, gold = apply_cdc(
        spark,
        events,
        base,
    )

    mark_completed(
        spark,
        base,
        batch_id,
        source_hash,
    )

    assert completed_batch(
        spark, base, batch_id, source_hash
    )

    bronze_path = f"{base}/bronze/events"
    silver_path = f"{base}/silver/orders"
    gold_path = f"{base}/gold/product_revenue"

    assert DeltaTable.isDeltaTable(
        spark, bronze_path
    )
    assert DeltaTable.isDeltaTable(
        spark, silver_path
    )
    assert DeltaTable.isDeltaTable(
        spark, gold_path
    )

    bronze = spark.read.format("delta").load(
        bronze_path
    )

    assert bronze.count() == 1
    assert silver.count() == 1

    order = silver.collect()[0]

    assert order["order_id"] == 101
    assert order["quantity"] == 2
    assert order["is_deleted"] is False
    assert order["revenue"] == pytest.approx(200.0)

    result = gold.collect()

    assert len(result) == 1
    assert result[0]["product"] == "Laptop"
    assert result[0]["total_revenue"] == pytest.approx(
        200.0
    )


def test_recovery_after_update_failure(spark, tmp_path):
    """An interrupted update must recover without double counting."""
    base = str(tmp_path / "update_recovery")

    initial = make_events(
        spark,
        [event("evt-001")],
    )

    apply_cdc(spark, initial, base)

    updated = make_events(
        spark,
        [
            event(
                "evt-002",
                quantity=5,
                operation="U",
                sequence=2,
            )
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated CDC failure after silver",
    ):
        apply_cdc(
            spark,
            updated,
            base,
            fail_after="silver",
        )

    silver, gold = apply_cdc(
        spark,
        updated,
        base,
    )

    bronze = spark.read.format("delta").load(
        f"{base}/bronze/events"
    )

    assert bronze.count() == 2

    order = silver.collect()[0]

    assert order["sequence_num"] == 2
    assert order["quantity"] == 5
    assert order["revenue"] == pytest.approx(500.0)

    assert gold.collect()[0]["total_revenue"] == pytest.approx(
        500.0
    )
