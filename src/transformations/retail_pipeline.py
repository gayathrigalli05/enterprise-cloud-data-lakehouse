
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

from src.quality.data_quality import validate_data

ROOT = Path(__file__).resolve().parents[2]
LAKEHOUSE = ROOT / "data/output/lakehouse"


def main():
    spark = (
        SparkSession.builder
        .appName("EnterpriseRetailLakehouse")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        # BRONZE: Ingest raw source data
        bronze = spark.read.csv(
            str(ROOT / "data/sample/orders.csv"),
            header=True,
            inferSchema=True
        )

        bronze_path = LAKEHOUSE / "bronze/orders"

        bronze.write.mode("overwrite").parquet(
            str(bronze_path)
        )

        print("Bronze layer saved successfully")

        # DATA QUALITY: Validate raw transactions
        quality_report = validate_data(bronze)

        if quality_report["total_records"] == 0:
            raise ValueError("Input dataset is empty")

        # SILVER: Clean and standardize
        silver = (
            bronze
            .dropDuplicates(["order_id"])
            .dropna(
                subset=[
                    "order_id",
                    "product",
                    "quantity",
                    "unit_price"
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

        silver_path = LAKEHOUSE / "silver/orders"

        silver.write.mode("overwrite").parquet(
            str(silver_path)
        )

        print("Silver layer saved successfully")

        # GOLD: Build business analytics
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

        gold_path = LAKEHOUSE / "gold/product_revenue"

        gold.write.mode("overwrite").parquet(
            str(gold_path)
        )

        print("Gold layer saved successfully")
        gold.show()

        print("Lakehouse pipeline completed!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
