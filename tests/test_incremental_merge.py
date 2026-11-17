
import pytest

from pyspark.sql import SparkSession
from delta.tables import DeltaTable

from src.transformations.incremental_merge import (
    merge_orders,
    deduplicate
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .appName("DeltaMergeTests")
        .master("local[2]")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    yield session
    session.stop()


def test_insert_update_and_rerun(spark, tmp_path):
    target = str(tmp_path / "orders")

    initial = spark.createDataFrame(
        [
            (1001, "C101", "Laptop", 1, 850.0),
            (1002, "C102", "Keyboard", 2, 45.0)
        ],
        [
            "order_id", "customer_id", "product",
            "quantity", "unit_price"
        ]
    )

    merge_orders(spark, initial, target)

    updates = spark.createDataFrame(
        [
            (1001, "C101", "Laptop", 2, 850.0),
            (1003, "C103", "Mouse", 3, 20.0)
        ],
        initial.schema
    )

    merge_orders(spark, updates, target)
    merge_orders(spark, updates, target)

    rows = {
        row.order_id: row
        for row in DeltaTable.forPath(
            spark, target
        ).toDF().collect()
    }

    assert len(rows) == 3
    assert rows[1001].quantity == 2
    assert rows[1002].quantity == 2
    assert rows[1003].quantity == 3


def test_conflicting_duplicates_rejected(spark):
    records = spark.createDataFrame(
        [
            (1001, "Laptop", 1),
            (1001, "Laptop", 2)
        ],
        ["order_id", "product", "quantity"]
    )

    with pytest.raises(ValueError):
        deduplicate(records)
