
from pyspark.sql.functions import col, count, when


def validate_data(df):
    """Validate retail transaction data."""

    total_records = df.count()

    null_counts = df.select([
        count(
            when(col(c).isNull(), 1)
        ).alias(c)
        for c in df.columns
    ]).collect()[0].asDict()

    duplicate_records = (
        df.groupBy("order_id")
        .count()
        .filter(col("count") > 1)
        .count()
    )

    invalid_transactions = df.filter(
        (col("quantity") <= 0)
        | (col("unit_price") < 0)
        | col("quantity").isNull()
        | col("unit_price").isNull()
    ).count()

    report = {
        "total_records": total_records,
        "null_counts": null_counts,
        "duplicate_order_ids": duplicate_records,
        "invalid_transactions": invalid_transactions,
    }

    print("DATA QUALITY REPORT")

    for key, value in report.items():
        print(f"{key}: {value}")

    return report
