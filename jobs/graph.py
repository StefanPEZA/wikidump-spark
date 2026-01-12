from pyspark.sql import functions as F
from pyspark.sql import DataFrame


def _norm_title(col: F.Column) -> F.Column:
    """
    Normalize titles for joining:
    - cast to string
    - trim
    - underscores -> spaces
    - collapse whitespace
    - lowercase
    """
    return F.lower(
        F.regexp_replace(
            F.regexp_replace(F.trim(col.cast("string")), r"_+", " "),
            r"\s+",
            " ",
        )
    )


def run_graph(df: DataFrame, out_base: str = None):
    print(">>> [4/4] Graph: Extracting Edges...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    if out_base is None:
        print(">>> ERROR: out_base is required (pass args.out from main.py).")
        return

    required_cols = {"id", "title"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f">>> ERROR: Missing required columns: {missing}. Got: {df.columns}")
        return

    # Prefer parser-produced array column "links"
    if "links" not in df.columns:
        print(">>> ERROR: Column 'links' not found. Parser should create it.")
        print(">>> Hint: ensure run_parser adds .withColumn('links', ...) as in your parser.py.")
        return

    # Filter to articles and non-redirect if present
    base = df
    if "ns" in base.columns:
        base = base.filter(F.col("ns").cast("int") == 0)
    if "redirect" in base.columns:
        base = base.filter(F.col("redirect").isNull())

    # Vertices (unique id/title)
    vertices = (
        base.select(F.col("id").cast("long").alias("id"),
                    F.col("title").cast("string").alias("title"))
        .dropna(subset=["id", "title"])
        .dropDuplicates(["id"])
    )

    # For joining by title
    v_index = vertices.select(
        F.col("id").alias("dst_id"),
        F.col("title").alias("dst_title"),
        _norm_title(F.col("title")).alias("dst_title_norm")
    )

    # Explode links and normalize
    exploded = (
        base.select(
            F.col("id").cast("long").alias("src_id"),
            F.col("title").cast("string").alias("src_title"),
            F.explode_outer(F.col("links")).alias("link_title_raw")
        )
        .dropna(subset=["src_id"])
    )

    exploded = (
        exploded
        .withColumn("link_title_raw", F.col("link_title_raw").cast("string"))
        .withColumn("link_title_norm", _norm_title(F.col("link_title_raw")))
        .filter(F.col("link_title_norm").isNotNull() & (F.col("link_title_norm") != ""))
    )

    # Join link title -> destination id
    edges_joined = (
        exploded.join(
            v_index,
            on=F.col("link_title_norm") == F.col("dst_title_norm"),
            how="inner"
        )
        .select("src_id", "dst_id")
        .filter(F.col("src_id") != F.col("dst_id"))  # remove self loops
    )

    # Weight: multiple mentions of same link in same page -> count
    edges = (
        edges_joined
        .groupBy("src_id", "dst_id")
        .count()
        .withColumnRenamed("count", "weight")
    )

    # Basic sanity stats
    v_cnt = vertices.count()
    e_cnt = edges.count()
    print(f">>> Graph vertices: {v_cnt:,}")
    print(f">>> Graph edges:    {e_cnt:,}")

    # Top by inlinks (sum weights)
    inlinks_top = (
        edges.groupBy("dst_id")
        .agg(F.sum("weight").alias("in_weight"), F.count(F.lit(1)).alias("in_degree"))
        .join(vertices.select(F.col("id").alias("dst_id"), F.col("title")), on="dst_id", how="left")
        .orderBy(F.desc("in_weight"), F.desc("in_degree"))
        .select(F.col("title").alias("page"), "dst_id", "in_weight", "in_degree")
        .limit(200)
    )

    vertices.write.mode("overwrite").parquet(f"{out_base}/graph/vertices")
    edges.write.mode("overwrite").parquet(f"{out_base}/graph/edges")
    inlinks_top.write.mode("overwrite").parquet(f"{out_base}/graph/inlinks_top")

    print(f">>> Wrote: {out_base}/graph/vertices")
    print(f">>> Wrote: {out_base}/graph/edges")
    print(f">>> Wrote: {out_base}/graph/inlinks_top")
