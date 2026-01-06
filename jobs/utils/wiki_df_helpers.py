from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, ArrayType


def has_path(schema, path: str) -> bool:
    """
    Check whether the DataFrame schema contains a nested path like "a.b.c".
    Works even when the schema includes arrays (it descends into elementType).
    """
    parts = path.split(".")
    cur = schema

    for p in parts:
        while isinstance(cur, ArrayType):
            cur = cur.elementType

        if not isinstance(cur, StructType):
            return False

        field = next((f for f in cur.fields if f.name == p), None)
        if field is None:
            return False

        cur = field.dataType

    return True


def _pick_col(df, candidates):
    """Return the first matching nested column from candidates, else None."""
    
    for c in candidates:
        if has_path(df.schema, c):
            return F.col(c)
    return None


def pick_text_col(df, candidates=None):
    """
    Pick the text column (as a pyspark.sql.Column) from common Wikipedia XML variants.
    Returns a Column if found, otherwise None.
    """
    if candidates is None:
        # Prefer the cleaned/flattened schema produced by our parser, but keep
        # backward compatibility with raw Wikipedia XML variants.
        candidates = ["text", "text_raw", "revision.text._VALUE", "revision.text"]
    return _pick_col(df, candidates)


def split_count(col, pattern: str):
    """
    Count occurrences of a pattern in a column using split:
    if the pattern appears N times, split produces N+1 parts => N = size(split) - 1.
    """
    return F.greatest(F.lit(0), F.size(F.split(col, pattern)) - 1)


def print_len_summary(df, label: str, len_col: str = "text_len"):
    """
    Print a statistical summary for a length column (defaults to "text_len").
    """
    print(f"\n>>> Text length summary ({label}):")
    (df.select(
        F.min(len_col).alias("min"),
        F.expr(f"percentile_approx({len_col}, 0.5)").alias("p50"),
        F.expr(f"percentile_approx({len_col}, 0.9)").alias("p90"),
        F.expr(f"percentile_approx({len_col}, 0.99)").alias("p99"),
        F.max(len_col).alias("max"),
        F.avg(len_col).alias("avg"),
        F.stddev(len_col).alias("stddev"),
    ).show(truncate=False))
