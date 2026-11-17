
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

from src.quality.data_quality import validate_data

ROOT = Path(__file__).resolve().parents[2]


def main():
    spark = (
        SparkSession.builder
        .appName("EnterpriseRetailLakehouse")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        # BRONZE: Ingest raw transactions
        bronze = spark.read.csv(
            str(ROOT / "data/sample/orders.csv"),
            header=True,
            inferSchema=True,
        )

        print("BRONZE LAYER")
        bronze.show()

        # DATA QUALITY: Validate raw records
        quality_report = validate_data(bronze)

        if quality_report["total_records"] == 0:
            raise ValueError("Input dataset is empty")

        print("Data quality validation completed")

        # SILVER: Clean and validate records
        silver = (
            bronze
            .dropDuplicates(["order_id"])
            .dropna(
                subset=[
                    "order_id",
                    "product",
                    "quantity",
                    "unit_price",
                ]
            )
            .filter(
                (col("quantity") > 0)
                & (col("unit_price") >= 0)
            )
            .withColumn(
                "revenue",
                col("quantity") * col("unit_price")
            )
        )

        print("SILVER LAYER")
        silver.show()

        # GOLD: Aggregate product revenue
        gold = (
            silver
            .groupBy("product")
            .agg(
                spark_sum("revenue").alias(
                    "total_revenue"
                )
            )
            .orderBy(col("total_revenue").desc())
        )

        print("GOLD LAYER")
        gold.show()

        # Write analytics-ready data
        output_path = ROOT / "data/output/gold_revenue"

        gold.write.mode("overwrite").parquet(
            str(output_path)
        )

        print("ETL pipeline completed successfully!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
