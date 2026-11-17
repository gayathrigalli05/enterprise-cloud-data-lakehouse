
import pytest

from pyspark.sql import functions as F

from src.transformations.delta_pipeline import create_spark
from src.transformations.cdc_pipeline import apply_cdc


@pytest.fixture(scope="module")
def spark():
    session = create_spark()
    yield session
    session.stop()


def events_df(spark, rows):
    return spark.createDataFrame(
        rows,
        [
            "event_id", "order_id", "customer_id",
            "product", "quantity", "unit_price",
            "operation", "sequence_num", "updated_at"
        ]
    )


def test_insert_update_delete(spark, tmp_path):
    events = events_df(spark, [
        ("E1", 1, "C1", "Laptop", 1, 100.0,
         "I", 1, "2026-11-17 09:00:00"),
        ("E2", 2, "C2", "Mouse", 2, 20.0,
         "I", 2, "2026-11-17 09:05:00"),
        ("E3", 1, "C1", "Laptop", 3, 100.0,
         "U", 3, "2026-11-17 10:00:00"),
        ("E4", 2, "C2", "Mouse", 2, 20.0,
         "D", 4, "2026-11-17 10:05:00"),
    ])

    silver, gold = apply_cdc(
        spark, events, tmp_path / "cdc"
    )

    result = {
        row.order_id: row
        for row in silver.collect()
    }

    assert len(result) == 2
    assert result[1].quantity == 3
    assert result[1].is_deleted is False
    assert result[2].is_deleted is True

    revenue = {
        row.product: row.total_revenue
        for row in gold.collect()
    }

    assert revenue == {"Laptop": 300.0}


def test_idempotent_replay(spark, tmp_path):
    events = events_df(spark, [
        ("E1", 1, "C1", "Laptop", 1, 100.0,
         "I", 1, "2026-11-17 09:00:00")
    ])

    path = tmp_path / "cdc"

    apply_cdc(spark, events, path)
    silver, gold = apply_cdc(spark, events, path)

    bronze = spark.read.format("delta").load(
        str(path / "bronze/events")
    )

    assert bronze.count() == 1
    assert silver.count() == 1
    assert gold.first().total_revenue == 100.0


def test_late_event_does_not_override(spark, tmp_path):
    path = tmp_path / "cdc"

    newer = events_df(spark, [
        ("E10", 1, "C1", "Laptop", 5, 100.0,
         "U", 10, "2026-11-17 12:00:00")
    ])

    older = events_df(spark, [
        ("E5", 1, "C1", "Laptop", 1, 100.0,
         "I", 5, "2026-11-17 09:00:00")
    ])

    apply_cdc(spark, newer, path)
    silver, gold = apply_cdc(spark, older, path)

    assert silver.first().quantity == 5
    assert gold.first().total_revenue == 500.0


def test_delete_prevents_old_resurrection(
    spark, tmp_path
):
    path = tmp_path / "cdc"

    deleted = events_df(spark, [
        ("E20", 1, "C1", "Laptop", 1, 100.0,
         "D", 20, "2026-11-17 12:00:00")
    ])

    older = events_df(spark, [
        ("E10", 1, "C1", "Laptop", 1, 100.0,
         "I", 10, "2026-11-17 09:00:00")
    ])

    apply_cdc(spark, deleted, path)
    silver, gold = apply_cdc(spark, older, path)

    assert silver.first().is_deleted is True
    assert gold.count() == 0


def test_invalid_operation_rejected(spark, tmp_path):
    invalid = events_df(spark, [
        ("E1", 1, "C1", "Laptop", 1, 100.0,
         "X", 1, "2026-11-17 09:00:00")
    ])

    with pytest.raises(ValueError):
        apply_cdc(
            spark, invalid, tmp_path / "invalid"
        )


def test_conflicting_sequence_rejected(
    spark, tmp_path
):
    invalid = events_df(spark, [
        ("E1", 1, "C1", "Laptop", 1, 100.0,
         "I", 1, "2026-11-17 09:00:00"),
        ("E2", 1, "C1", "Laptop", 2, 100.0,
         "U", 1, "2026-11-17 10:00:00"),
    ])

    with pytest.raises(ValueError):
        apply_cdc(
            spark, invalid, tmp_path / "conflict"
        )
