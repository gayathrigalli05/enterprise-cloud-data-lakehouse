
import hashlib

import pytest
from delta.tables import DeltaTable
from pyspark.sql import functions as F

from src.transformations.cdc_pipeline import apply_cdc
from src.transformations.cdc_checkpoint import (
    checkpoint_path,
    completed_batch,
    mark_completed,
)


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


def make_events(spark, rows):
    """Create CDC events with a consistent schema."""
    return spark.createDataFrame(
        rows,
        schema=(
            "event_id string, order_id long, "
            "customer_id string, product string, "
            "quantity int, unit_price double, "
            "operation string, sequence_num long, "
            "updated_at string"
        ),
    )


def event(
    event_id,
    order_id=101,
    product="Laptop",
    quantity=2,
    price=100.0,
    operation="I",
    sequence=1,
):
    return (
        event_id,
        order_id,
        "C001",
        product,
        quantity,
        price,
        operation,
        sequence,
        "2026-01-01 10:00:00",
    )


def test_identical_event_replay(spark, tmp_path):
    """Replaying an event must not duplicate Bronze data."""
    base = str(tmp_path / "replay")

    events = make_events(
        spark, [event("evt-001")]
    )

    apply_cdc(spark, events, base)
    silver, gold = apply_cdc(spark, events, base)

    bronze = spark.read.format("delta").load(
        f"{base}/bronze/events"
    )

    assert bronze.count() == 1
    assert silver.count() == 1

    revenue = gold.collect()[0]["total_revenue"]
    assert revenue == pytest.approx(200.0)


def test_conflicting_event_id(spark, tmp_path):
    """The same event ID cannot represent changed data."""
    base = str(tmp_path / "event_conflict")

    apply_cdc(
        spark,
        make_events(spark, [event("evt-001")]),
        base,
    )

    conflicting = make_events(
        spark,
        [event("evt-001", quantity=5)],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting CDC event_id",
    ):
        apply_cdc(spark, conflicting, base)

    bronze = spark.read.format("delta").load(
        f"{base}/bronze/events"
    )

    assert bronze.count() == 1


def test_conflicting_order_sequence(spark, tmp_path):
    """Different events cannot share an order sequence."""
    base = str(tmp_path / "sequence_conflict")

    apply_cdc(
        spark,
        make_events(spark, [event("evt-001")]),
        base,
    )

    conflicting = make_events(
        spark,
        [event("evt-002", quantity=3, sequence=1)],
    )

    with pytest.raises(
        ValueError,
        match="Conflicting order sequence",
    ):
        apply_cdc(spark, conflicting, base)


def test_out_of_order_event(spark, tmp_path):
    """An older event must not overwrite newer order state."""
    base = str(tmp_path / "out_of_order")

    newer = make_events(
        spark,
        [event("evt-002", quantity=5, sequence=2)],
    )

    older = make_events(
        spark,
        [event("evt-001", quantity=2, sequence=1)],
    )

    apply_cdc(spark, newer, base)
    silver, gold = apply_cdc(spark, older, base)

    row = silver.collect()[0]

    assert row["sequence_num"] == 2
    assert row["quantity"] == 5
    assert gold.collect()[0]["total_revenue"] == pytest.approx(
        500.0
    )


def test_soft_delete(spark, tmp_path):
    """Deleted orders remain in Silver but leave Gold."""
    base = str(tmp_path / "soft_delete")

    apply_cdc(
        spark,
        make_events(spark, [event("evt-001")]),
        base,
    )

    delete = make_events(
        spark,
        [
            event(
                "evt-002",
                quantity=None,
                price=None,
                product=None,
                operation="D",
                sequence=2,
            )
        ],
    )

    silver, gold = apply_cdc(spark, delete, base)

    assert silver.count() == 1
    assert silver.collect()[0]["is_deleted"] is True
    assert gold.count() == 0


def test_checkpoint_persistence(spark, tmp_path):
    """A completed checkpoint must survive subsequent reads."""
    base = str(tmp_path / "checkpoint")
    batch_id = "batch-001"
    source_hash = hashlib.sha256(
        b"example source data"
    ).hexdigest()

    assert not completed_batch(
        spark, base, batch_id, source_hash
    )

    mark_completed(
        spark, base, batch_id, source_hash
    )

    assert completed_batch(
        spark, base, batch_id, source_hash
    )

    path = checkpoint_path(base)

    assert DeltaTable.isDeltaTable(spark, path)


def test_duplicate_checkpoint(spark, tmp_path):
    """Marking the same batch twice must not add a row."""
    base = str(tmp_path / "duplicate_checkpoint")

    mark_completed(spark, base, "batch-001", "hash-001")
    mark_completed(spark, base, "batch-001", "hash-001")

    checkpoints = spark.read.format("delta").load(
        checkpoint_path(base)
    )

    assert checkpoints.count() == 1


def test_changed_source_checkpoint(spark, tmp_path):
    """A completed batch ID cannot have different content."""
    base = str(tmp_path / "changed_source")

    mark_completed(spark, base, "batch-001", "hash-old")

    with pytest.raises(
        ValueError,
        match="different source content",
    ):
        completed_batch(
            spark, base, "batch-001", "hash-new"
        )


def test_recovery_after_interrupted_run(spark, tmp_path):
    """
    Simulate a crash after CDC data is written
    but before the completion checkpoint is saved.
    """
    base = str(tmp_path / "recovery")
    batch_id = "batch-001"
    source_hash = "hash-001"

    events = make_events(
        spark, [event("evt-001")]
    )

    # First run writes data but fails before checkpoint.
    apply_cdc(spark, events, base)

    assert not completed_batch(
        spark, base, batch_id, source_hash
    )

    # Recovery replays the same batch.
    silver, gold = apply_cdc(spark, events, base)

    mark_completed(
        spark, base, batch_id, source_hash
    )

    assert completed_batch(
        spark, base, batch_id, source_hash
    )

    bronze = spark.read.format("delta").load(
        f"{base}/bronze/events"
    )

    assert bronze.count() == 1
    assert silver.count() == 1
    assert gold.collect()[0]["total_revenue"] == pytest.approx(
        200.0
    )
