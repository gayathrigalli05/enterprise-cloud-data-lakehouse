
from pathlib import Path
import argparse

from delta.tables import DeltaTable
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, row_number, sum as spark_sum
)
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType
)

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_id", StringType(), True),
    StructField("product", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True)
])


def create_spark():
    return (
        SparkSession.builder
        .appName("IncrementalDeltaMerge")
        .master("local[2]")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension"
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )


def read_orders(spark, path):
    df = (
        spark.read
        .option("header", True)
        .schema(SCHEMA)
        .csv(str(path))
    )

    return df.filter(
        col("order_id").isNotNull()
        & col("quantity").isNotNull()
        & (col("quantity") > 0)
        & col("unit_price").isNotNull()
        & (col("unit_price") >= 0)
    )


def deduplicate(df):
    # For this sample, identical duplicate rows
    # are removed. Conflicting updates are rejected.
    unique = df.dropDuplicates()

    conflicts = (
        unique.groupBy("order_id")
        .count()
        .filter(col("count") > 1)
    )

    if conflicts.limit(1).count() > 0:
        raise ValueError(
            "Conflicting records for the same order_id"
        )

    return unique


def merge_orders(spark, incoming, target_path):
    incoming = deduplicate(incoming)
    target_path = str(target_path)

    if not DeltaTable.isDeltaTable(spark, target_path):
        (
            incoming.write
            .format("delta")
            .mode("errorifexists")
            .save(target_path)
        )
        return

    target = DeltaTable.forPath(spark, target_path)

    (
        target.alias("target")
        .merge(
            incoming.alias("source"),
            "target.order_id = source.order_id"
        )
        .whenMatchedUpdate(
            condition="""
                NOT (
                    target.customer_id <=> source.customer_id
                    AND target.product <=> source.product
                    AND target.quantity <=> source.quantity
                    AND target.unit_price <=> source.unit_price
                )
            """,
            set={
                "customer_id": "source.customer_id",
                "product": "source.product",
                "quantity": "source.quantity",
                "unit_price": "source.unit_price"
            }
        )
        .whenNotMatchedInsertAll()
        .execute()
    )


def build_gold(spark, silver_path, gold_path):
    silver = spark.read.format("delta").load(
        str(silver_path)
    )

    gold = (
        silver.withColumn(
            "revenue",
            col("quantity") * col("unit_price")
        )
        .groupBy("product")
        .agg(
            spark_sum("quantity").alias("total_quantity"),
            spark_sum("revenue").alias("total_revenue")
        )
    )

    (
        gold.write
        .format("delta")
        .mode("overwrite")
        .save(str(gold_path))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(
            ROOT / "data/sample/orders_incremental.csv"
        )
    )
    parser.add_argument(
        "--silver",
        default=str(ROOT / "data/delta/silver/orders")
    )
    parser.add_argument(
        "--gold",
        default=str(ROOT / "data/delta/gold/product_sales")
    )
    args = parser.parse_args()

    spark = create_spark()

    try:
        incoming = read_orders(spark, args.input)

        merge_orders(spark, incoming, args.silver)

        build_gold(spark, args.silver, args.gold)

        print("Incremental Delta MERGE completed")

        spark.read.format("delta").load(
            args.silver
        ).orderBy("order_id").show()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
