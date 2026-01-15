from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import os


def _as_non_empty_string(col: F.Column) -> F.Column:
    return F.when(F.trim(col.cast("string")) != "", F.trim(col.cast("string"))).otherwise(F.lit(None))


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


def _stable_node_id(prefix: str, title_col: F.Column) -> F.Column:
    """Generate a stable negative node id for non-page entities.

    We keep non-page nodes negative to avoid collisions with Wikipedia page ids.
    Prefix is included to avoid collisions across node kinds.
    """
    key = F.concat(F.lit(prefix), F.lit(":"), F.coalesce(title_col.cast("string"), F.lit("")))
    # xxhash64 returns a 64-bit signed integer; abs + negate makes it negative.
    return -F.abs(F.xxhash64(key)).cast("long")


def run_graph(
    df: DataFrame,
    out_base: str = None,
):
    print(">>> [4/4] Graph: Building Knowledge Graph...")

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

    spark = df.sparkSession

    # --- Page vertices (unique id/title) ---
    page_vertices = (
        base.select(
            F.col("id").cast("long").alias("id"),
            _as_non_empty_string(F.col("title")).alias("title"),
        )
        .dropna(subset=["id", "title"])
        .dropDuplicates(["id"])
        .withColumn("kind", F.lit("page"))
        .withColumn("title_norm", _norm_title(F.col("title")))
    )

    # For joining by title
    v_index = page_vertices.select(
        F.col("id").alias("dst_id"),
        F.col("title").alias("dst_title"),
        F.col("title_norm").alias("dst_title_norm"),
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
    page_link_edges = (
        edges_joined
        .groupBy("src_id", "dst_id")
        .count()
        .withColumnRenamed("count", "weight")
        .withColumn("edge_type", F.lit("link"))
    )

    # --- Category nodes/edges (page -> category) ---
    category_vertices = None
    category_edges = None
    if "categories" in base.columns:
        cats = (
            base.select(
                F.col("id").cast("long").alias("src_id"),
                F.explode_outer(F.col("categories")).alias("category_raw"),
            )
            .withColumn("category_raw", _as_non_empty_string(F.col("category_raw")))
            .dropna(subset=["src_id", "category_raw"])
            .withColumn("category_title", F.concat(F.lit("Category:"), F.col("category_raw")))
            .withColumn("dst_id", _stable_node_id("category", F.col("category_title")))
        )

        category_vertices = (
            cats.select(
                F.col("dst_id").cast("long").alias("id"),
                F.col("category_title").alias("title"),
            )
            .dropDuplicates(["id"])
            .withColumn("kind", F.lit("category"))
            .withColumn("title_norm", _norm_title(F.col("title")))
        )

        category_edges = (
            cats.select("src_id", "dst_id")
            .dropDuplicates(["src_id", "dst_id"])
            .withColumn("weight", F.lit(1))
            .withColumn("edge_type", F.lit("has_category"))
        )

    # --- Combine vertices/edges ---
    vertices = page_vertices
    if category_vertices is not None:
        vertices = vertices.unionByName(category_vertices, allowMissingColumns=True)

    edges = page_link_edges
    if category_edges is not None:
        edges = edges.unionByName(category_edges, allowMissingColumns=True)

    # Basic sanity stats
    v_cnt = vertices.count()
    e_cnt = edges.count()
    page_v_cnt = page_vertices.count()
    page_link_e_cnt = page_link_edges.count()

    print(f">>> Vertices total: {v_cnt:,} (pages: {page_v_cnt:,})")
    print(f">>> Edges total:    {e_cnt:,} (page links: {page_link_e_cnt:,})")

    # --- Clean schemas for output ---
    vertices_out = vertices.select(
        F.col("id").cast("long").alias("id"),
        F.col("title").cast("string").alias("title"),
        F.col("kind").cast("string").alias("kind"),
    )

    edges_out = edges.select(
        F.col("src_id").cast("long").alias("src_id"),
        F.col("dst_id").cast("long").alias("dst_id"),
        F.col("weight").cast("long").alias("weight"),
        F.col("edge_type").cast("string").alias("edge_type"),
    )

    # --- GraphFrames PageRank ---
    try:
        from graphframes import GraphFrame  # type: ignore

        max_iter = int(os.getenv("PAGERANK_MAX_ITER", "10"))
        top_n = int(os.getenv("PAGERANK_TOP_N", "10"))

        g = GraphFrame(
            vertices_out.select("id", "title", "kind"),
            edges_out.select(F.col("src_id").alias("src"), F.col("dst_id").alias("dst")),
        )

        pr = g.pageRank(resetProbability=0.15, maxIter=max_iter)
        print(">>> Top nodes by PageRank:")
        (
            pr.vertices.select("id", "title", "kind", F.col("pagerank").cast("double").alias("pagerank"))
            .where(F.col("kind") == "page")
            .orderBy(F.desc("pagerank"))
            .show(top_n, truncate=False)
        )
        (
            pr.vertices.select("id", "title", "kind", F.col("pagerank").cast("double").alias("pagerank"))
            .where(F.col("kind") == "category")
            .orderBy(F.desc("pagerank"))
            .show(top_n, truncate=False)
        )
    except Exception as e:
        print(">>> PageRank skipped (GraphFrames not available).")
        print(f">>> Details: {e}")
        
    # --- Write outputs ---
    (
        vertices_out
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"{out_base}/graph/nodes_csv")
    )
    (
        edges_out
        .coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(f"{out_base}/graph/edges_csv")
    )
    print(f">>> Wrote: {out_base}/graph/nodes_csv (csv)")
    print(f">>> Wrote: {out_base}/graph/edges_csv (csv)")

    print(">>> Graph stage complete")
