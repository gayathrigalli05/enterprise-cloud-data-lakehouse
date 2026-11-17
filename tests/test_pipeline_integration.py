
from pathlib import Path

from pyspark.sql import SparkSession

from src.transformations import retail_pipeline


def test_end_to_end_lakehouse_pipeline(tmp_path, monkeypatch):
    """
    Test the complete Bronze, Silver, Gold pipeline.
    """

    # Use temporary storage for test outputs
    monkeypatch.setattr(
        retail_pipeline,
        "LAKEHOUSE",
        tmp_path / "lakehouse"
    )

    # Run the actual ETL pipeline
    retail_pipeline.main()

    # Define output locations
    bronze_path = tmp_path / "lakehouse/bronze/orders"
    silver_path = tmp_path / "lakehouse/silver/orders"
    gold_path = tmp_path / "lakehouse/gold/product_revenue"

    # Confirm all three layers exist
    assert bronze_path.exists()
    assert silver_path.exists()
    assert gold_path.exists()

    # Read generated Parquet datasets
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("LakehouseIntegrationTest")
        .getOrCreate()
    )

    try:
        bronze = spark.read.parquet(str(bronze_path))
        silver = spark.read.parquet(str(silver_path))
        gold = spark.read.parquet(str(gold_path))

        # Validate record counts
        assert bronze.count() == 10
        assert silver.count() == 10
        assert gold.count() == 5

        # Validate revenue calculation
        results = {
            row["product"]: row["total_revenue"]
            for row in gold.collect()
        }

        assert results["Laptop"] == 1750
        assert results["Monitor"] == 500
        assert results["Keyboard"] == 255
        assert results["Headphones"] == 150
        assert results["Mouse"] == 110

    finally:
        spark.stop()
