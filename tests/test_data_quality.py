
import pytest

from pyspark.sql import SparkSession
from src.quality.data_quality import validate_data


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("DataQualityTests")
        .getOrCreate()
    )

    yield session
    session.stop()


def create_test_dataframe(spark, records):
    schema = (
        "order_id INT, customer_id STRING, "
        "product STRING, quantity INT, "
        "unit_price DOUBLE"
    )
    return spark.createDataFrame(records, schema=schema)


def test_valid_transactions(spark):
    records = [
        (1001, "C101", "Laptop", 1, 850.0),
        (1002, "C102", "Mouse", 2, 25.0),
    ]

    df = create_test_dataframe(spark, records)
    report = validate_data(df)

    assert report["total_records"] == 2
    assert report["duplicate_order_ids"] == 0
    assert report["invalid_transactions"] == 0


def test_duplicate_order_detection(spark):
    records = [
        (1001, "C101", "Laptop", 1, 850.0),
        (1001, "C101", "Laptop", 1, 850.0),
    ]

    df = create_test_dataframe(spark, records)
    report = validate_data(df)

    assert report["duplicate_order_ids"] == 1


def test_invalid_quantity_detection(spark):
    records = [
        (1001, "C101", "Laptop", -1, 850.0),
        (1002, "C102", "Mouse", 0, 25.0),
    ]

    df = create_test_dataframe(spark, records)
    report = validate_data(df)

    assert report["invalid_transactions"] == 2


def test_missing_values(spark):
    records = [
        (1001, "C101", None, 1, 850.0),
        (1002, "C102", "Mouse", None, 25.0),
    ]

    df = create_test_dataframe(spark, records)
    report = validate_data(df)

    assert report["null_counts"]["product"] == 1
    assert report["null_counts"]["quantity"] == 1
    assert report["invalid_transactions"] == 1
