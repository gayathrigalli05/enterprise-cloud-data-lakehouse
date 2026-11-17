
import pytest
from pyspark.sql import functions as F

from src.transformations.delta_pipeline import create_spark
from src.transformations.watermark_incremental import (
    merge_incremental,
    get_watermark,
    filter_new_records,
)


@pytest.fixture(scope="module")
def spark():
    session = create_spark()
    yield session
    session.stop()


def make_orders(spark, rows):
    return spark.createDataFrame(
        rows,
        [
            "order_id", "customer_id", "product",
            "quantity", "unit_price", "updated_at"
        ]
    )


def test_insert_update_and_idempotency(spark, tmp_path):
    path = tmp_path / "orders"

    initial = make_orders(spark, [
        (1001, "C101", "Laptop", 1, 850.0,
         "2026-11-17 09:00:00"),
        (1002, "C102", "Mouse", 2, 20.0,
         "2026-11-17 09:05:00"),
    ])

    merge_incremental(spark, initial, path)

    updates = make_orders(spark, [
        (1001, "C101", "Laptop", 3, 850.0,
         "2026-11-17 10:00:00"),
        (1003, "C103", "Keyboard", 1, 45.0,
         "2026-11-17 10:10:00"),
    ])

    merge_incremental(spark, updates, path)
    merge_incremental(spark, updates, path)

    result = {
        r.order_id: r
        for r in spark.read.format("delta")
        .load(str(path)).collect()
    }

    assert len(result) == 3
    assert result[1001].quantity == 3
    assert result[1002].quantity == 2
    assert result[1003].quantity == 1


def test_older_update_does_not_overwrite(spark, tmp_path):
    path = tmp_path / "orders"

    latest = make_orders(spark, [
        (1001, "C101", "Laptop", 5, 850.0,
         "2026-11-17 12:00:00")
    ])
    older = make_orders(spark, [
        (1001, "C101", "Laptop", 1, 850.0,
         "2026-11-17 08:00:00")
    ])

    merge_incremental(spark, latest, path)
    merge_incremental(spark, older, path)

    row = spark.read.format("delta").load(
        str(path)
    ).first()

    assert row.quantity == 5


def test_watermark_filter(spark, tmp_path):
    path = tmp_path / "orders"

    initial = make_orders(spark, [
        (1001, "C101", "Laptop", 1, 850.0,
         "2026-11-17 09:00:00")
    ])
    merge_incremental(spark, initial, path)

    watermark = get_watermark(spark, path)
    assert watermark is not None

    incoming = make_orders(spark, [
        (1002, "C102", "Mouse", 1, 20.0,
         "2026-11-17 08:00:00"),
        (1003, "C103", "Keyboard", 1, 45.0,
         "2026-11-17 11:00:00"),
    ])

    filtered = filter_new_records(
        incoming, watermark
    )

    assert [
        r.order_id for r in filtered.collect()
    ] == [1003]


def test_conflicting_timestamp_rejected(spark, tmp_path):
    incoming = make_orders(spark, [
        (1001, "C101", "Laptop", 1, 850.0,
         "2026-11-17 10:00:00"),
        (1001, "C101", "Laptop", 2, 850.0,
         "2026-11-17 10:00:00"),
    ])

    with pytest.raises(ValueError):
        merge_incremental(
            spark, incoming, tmp_path / "orders"
        )
