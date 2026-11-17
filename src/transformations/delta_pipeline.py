
from pathlib import Path

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    sum as spark_sum,
    row_number,
)
from pyspark.sql.window import Window

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


def merge_delta(spark, dataframe, path, key):
    path = str(path)

    if DeltaTable.isDeltaTable(spark, path):
        target = DeltaTable.forPath(spark, path)

        (
            target.alias("target")
            .merge(
                dataframe.alias("source"),
                f"target.{key} = source.{key}"
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        (
            dataframe.write
            .format("delta")
            .mode("errorifexists")
            .save(path)
        )


def main():
    spark = create_spark()

    try:
        source = spark.read.csv(
            str(ROOT / "data/sample/orders.csv"),
            header=True,
            inferSchema=True
        )

        report = validate_data(source)

        if report["total_records"] == 0:
            raise ValueError("Input dataset is empty")

        # Bronze: retain the latest incoming row per order.
        # This assumes order_id identifies an order.
        bronze = source.dropDuplicates(["order_id"])

        merge_delta(
            spark,
            bronze,
            LAKEHOUSE / "bronze/orders",
            "order_id"
        )

        # Silver: transform the complete current Bronze state.
        bronze_current = (
            spark.read.format("delta")
            .load(str(LAKEHOUSE / "bronze/orders"))
        )

        silver = (
            bronze_current
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

        merge_delta(
            spark,
            silver,
            LAKEHOUSE / "silver/orders",
            "order_id"
        )

        # Gold: aggregate the full current Silver state.
        silver_current = (
            spark.read.format("delta")
            .load(str(LAKEHOUSE / "silver/orders"))
        )

        gold = (
            silver_current
            .groupBy("product")
            .agg(
                spark_sum("revenue").alias(
                    "total_revenue"
                )
            )
        )

        # Replace Gold because products can disappear
        # or have their totals reduced after updates.
        (
            gold.write
            .format("delta")
            .mode("overwrite")
            .save(str(LAKEHOUSE / "gold/product_revenue"))
        )

        print("Incremental Delta MERGE completed")
        gold.show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
