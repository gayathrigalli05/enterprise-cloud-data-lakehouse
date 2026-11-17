
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum as spark_sum,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    spark = (
        SparkSession.builder
        .appName("EnterpriseRetailLakehouse")
        .master("local[*]")
        .getOrCreate()
    )

    try:
        # BRONZE: Load raw retail transactions
        bronze = spark.read.csv(
            str(ROOT / "data/sample/orders.csv"),
            header=True,
            inferSchema=True,
        )

        print("BRONZE LAYER")
        bronze.show()

        # SILVER: Clean and validate transactions
        silver = (
            bronze
            .dropDuplicates(["order_id"])
            .dropna(subset=["order_id", "product"])
            .filter(
                (col("quantity") > 0)
                & (col("unit_price") >= 0)
            )
            .withColumn(
                "revenue",
                col("quantity") * col("unit_price"),
            )
        )

        print("SILVER LAYER")
        silver.show()

        # GOLD: Calculate revenue by product
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

        # Save analytics-ready results
        output_path = ROOT / "data/output/gold_revenue"

        gold.write.mode("overwrite").parquet(
            str(output_path)
        )

        print("Pipeline completed successfully!")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
