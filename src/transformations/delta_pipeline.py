
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum

from src.quality.data_quality import validate_data

ROOT = Path(__file__).resolve().parents[2]
LAKEHOUSE = ROOT / "data/output/delta_lakehouse"


def create_spark():
    builder = (
        SparkSession.builder
        .appName("EnterpriseDeltaLakehouse")
        .master("local[2]")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
    )

    return configure_spark_with_delta_pip(
        builder
    ).getOrCreate()


def main():
    spark = create_spark()

    try:
        bronze = spark.read.csv(
            str(ROOT / "data/sample/orders.csv"),
            header=True,
            inferSchema=True
        )

        bronze.write.format("delta").mode(
            "overwrite"
        ).save(str(LAKEHOUSE / "bronze/orders"))

        report = validate_data(bronze)

        if report["total_records"] == 0:
            raise ValueError("Input dataset is empty")

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

        silver.write.format("delta").mode(
            "overwrite"
        ).save(str(LAKEHOUSE / "silver/orders"))

        gold = (
            silver.groupBy("product")
            .agg(
                spark_sum("revenue").alias(
                    "total_revenue"
                )
            )
        )

        gold.write.format("delta").mode(
            "overwrite"
        ).save(str(LAKEHOUSE / "gold/product_revenue"))

        print("Delta Lake pipeline completed")
        gold.show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
