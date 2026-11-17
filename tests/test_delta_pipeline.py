
from delta.tables import DeltaTable

from src.transformations import delta_pipeline


def test_delta_lakehouse_pipeline(tmp_path, monkeypatch):
    # Redirect output to temporary test directory
    lakehouse = tmp_path / "delta_lakehouse"

    monkeypatch.setattr(
        delta_pipeline,
        "LAKEHOUSE",
        lakehouse
    )

    # Run Delta Lake ETL
    delta_pipeline.main()

    # Create Spark session to inspect outputs
    spark = delta_pipeline.create_spark()

    try:
        bronze_path = lakehouse / "bronze/orders"
        silver_path = lakehouse / "silver/orders"
        gold_path = lakehouse / "gold/product_revenue"

        # Verify all Delta tables exist
        for path in [
            bronze_path,
            silver_path,
            gold_path
        ]:
            assert DeltaTable.isDeltaTable(
                spark, str(path)
            )

        # Read Delta tables
        bronze = spark.read.format("delta").load(
            str(bronze_path)
        )
        silver = spark.read.format("delta").load(
            str(silver_path)
        )
        gold = spark.read.format("delta").load(
            str(gold_path)
        )

        # Verify row counts
        assert bronze.count() == 10
        assert silver.count() == 10
        assert gold.count() == 5

        # Verify business revenue
        revenue = {
            row["product"]: row["total_revenue"]
            for row in gold.collect()
        }

        assert revenue["Laptop"] == 1750
        assert revenue["Monitor"] == 500
        assert revenue["Keyboard"] == 255
        assert revenue["Headphones"] == 150
        assert revenue["Mouse"] == 110

    finally:
        spark.stop()
