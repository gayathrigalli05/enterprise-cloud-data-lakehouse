
from pathlib import Path

from pyspark.sql import functions as F

from src.transformations.delta_pipeline import create_spark
from src.transformations.cdc_pipeline import apply_cdc

ROOT = Path(__file__).resolve().parents[2]

CDC_SOURCE = ROOT / "data/sample/orders_cdc.csv"
CDC_LAKEHOUSE = ROOT / "data/output/cdc_lakehouse"


def main():
    spark = create_spark()

    try:
        events = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(str(CDC_SOURCE))
        )

        silver, gold = apply_cdc(
            spark,
            events,
            CDC_LAKEHOUSE
        )

        print("CDC pipeline completed successfully")

        print("Current Silver order states:")
        silver.orderBy("order_id").show(
            truncate=False
        )

        print("Active order count:")
        print(
            silver.filter(
                ~F.col("is_deleted")
            ).count()
        )

        print("Gold product revenue:")
        gold.orderBy("product").show(
            truncate=False
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
