from __future__ import annotations
from datetime import datetime
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

from jobs.utils.wiki_df_helpers import (
    pick_text_col,
    split_count,
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _as_int(x) -> int:
    try:
        return int(x) if x is not None else 0
    except Exception:
        return 0


def _as_float(x) -> float:
    try:
        return float(x) if x is not None else 0.0
    except Exception:
        return 0.0


def _maybe_plot_bar_png(
    png_path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    rotate_xticks: bool = True,
) -> bool:
    """Try to write a simple bar chart; returns False if matplotlib is unavailable."""
    try:
        import importlib

        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=160)
    plt.close()
    return True


def _write_stats_outputs(
    *,
    out_dir: Path,
    overview_df: DataFrame,
    ns_dist_df: DataFrame | None,
    global_len_df: DataFrame | None,
    articles_len_df: DataFrame | None,
    link_summary_df: DataFrame,
    top_most_linked_df: DataFrame | None,
    top_tokens_df: DataFrame,
) -> None:
    # 1) Single-row overview (parquet only)
    overview_df.write.mode("overwrite").parquet(str(out_dir / "overview"))

    # 2) Tables
    if ns_dist_df is not None:
        ns_dist_df.write.mode("overwrite").parquet(str(out_dir / "namespace_distribution"))

        # Optional bar chart (top 20 namespaces)
        top_ns = ns_dist_df.orderBy(F.desc("count")).limit(20).collect()
        labels = [str(r["ns"]) for r in top_ns]
        vals = [float(r["count"]) for r in top_ns]
        _maybe_plot_bar_png(
            out_dir / "namespace_distribution_top20.png",
            labels,
            vals,
            title="Namespace distribution (top 20)",
            xlabel="Namespace (ns)",
            ylabel="Pages",
        )

    # 3) Length summaries as 1-row tables
    if global_len_df is not None:
        global_len_df.write.mode("overwrite").parquet(str(out_dir / "text_length_summary_global"))
    if articles_len_df is not None:
        articles_len_df.write.mode("overwrite").parquet(str(out_dir / "text_length_summary_articles"))

    # 4) Link density + top pages
    link_summary_df.write.mode("overwrite").parquet(str(out_dir / "link_density_summary"))

    if top_most_linked_df is not None:
        top_most_linked_df.write.mode("overwrite").parquet(str(out_dir / "top_most_linked"))

    # 5) Vocabulary table
    top_tokens_df.write.mode("overwrite").parquet(str(out_dir / "top_tokens"))

    # Optional bar chart: top tokens
    try:
        top_tokens = top_tokens_df.orderBy(F.desc("count")).limit(25).collect()
        labels = [str(r["token"]) for r in top_tokens]
        vals = [float(r["count"]) for r in top_tokens]
        _maybe_plot_bar_png(
            out_dir / "top_tokens_top25.png",
            labels,
            vals,
            title="Top tokens (top 25)",
            xlabel="Token",
            ylabel="Count",
            rotate_xticks=True,
        )
    except Exception:
        pass


def run_stats(df: DataFrame, out_dir: str | None = None, top_k: int = 30) -> None:
    """Compute and print dataset statistics.
    """

    # Print the stage header immediately so users see progress before Spark actions.
    print(">>> [2/4] EDA: Statistics...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    out_path = Path(out_dir)

    # Console tables are capped at 25 rows for readability.
    top_k = int(min(int(top_k), 25))

    has_title = "title" in df.columns
    has_ns = "ns" in df.columns
    has_redirect = "redirect" in df.columns

    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # 1) Basic counts (global)
    # NOTE: prefer aggregations over repeated full scans.
    agg_exprs = [F.count(F.lit(1)).alias("total_pages")]
    if has_redirect:
        agg_exprs.append(F.sum(F.when(F.col("redirect").isNotNull(), 1).otherwise(0)).alias("redirect_pages"))

    total_row = df.agg(*agg_exprs).first()
    total = _as_int(total_row["total_pages"])
    redirects_global = _as_int(total_row["redirect_pages"]) if has_redirect else 0
    if has_redirect:
        print(f">>> pages={total:,}  redirects={redirects_global:,} ({_safe_ratio(redirects_global, total):.2%})")
    else:
        print(f">>> pages={total:,}")

    distinct_titles = 0
    missing_titles = 0
    dup_titles = 0
    if has_title:
        title_col = F.col("title")
        title_stats = (
            df.select(
                F.countDistinct("title").alias("distinct_titles"),
                F.sum(
                    F.when(
                        title_col.isNull() | (F.trim(title_col) == ""),
                        F.lit(1),
                    ).otherwise(F.lit(0))
                ).alias("missing_titles"),
            )
            .first()
        )
        distinct_titles = _as_int(title_stats["distinct_titles"])
        missing_titles = _as_int(title_stats["missing_titles"])
        dup_titles = df.groupBy("title").count().filter(F.col("count") > 1).count()
    if has_title:
        print(f">>> titles distinct={distinct_titles:,}  missing={missing_titles:,}  dup_groups={dup_titles:,}")

    ns_dist_df = df.groupBy("ns").count() if has_ns else None

    if has_ns and ns_dist_df is not None:
        print(">>> namespaces (top 25):")
        ns_dist_df.orderBy(F.desc("count")).show(25, truncate=False)

    # 2) Choose text column
    # Prefer cleaned parser output, but keep backward compatibility with raw XML variants.
    text_col = pick_text_col(df)

    if text_col is None:
        print(">>> ERROR: Text column not found. Skipping text-based stats.")
        return

    # keep original df intact; use an internal column for stats
    df_base = df.withColumn("text_stats", F.coalesce(text_col.cast("string"), F.lit("")))

    # 3) Global text stats
    df_global = df_base.withColumn("text_len", F.length(F.col("text_stats"))).cache()

    global_len_row = (
        df_global.agg(
            F.sum(F.when(F.length(F.trim(F.col("text_stats"))) == 0, 1).otherwise(0)).alias("empty_text"),
            F.min("text_len").alias("min"),
            F.expr("percentile_approx(text_len, 0.5)").alias("p50"),
            F.expr("percentile_approx(text_len, 0.9)").alias("p90"),
            F.expr("percentile_approx(text_len, 0.99)").alias("p99"),
            F.max("text_len").alias("max"),
            F.avg("text_len").alias("avg"),
            F.stddev("text_len").alias("stddev"),
        )
        .first()
    )

    empty_text = _as_int(global_len_row["empty_text"])
    print(f">>> empty_text_global={empty_text:,} ({_safe_ratio(empty_text, total):.2%})")


    # 4) Focused dataset: articles (ns=0) and non-redirect
    df_articles = df_global
    if has_ns:
        df_articles = df_articles.filter(F.col("ns").cast("int") == 0)
    if has_redirect:
        df_articles = df_articles.filter(F.col("redirect").isNull())

    articles_len_row = (
        df_articles.agg(
            F.count(F.lit(1)).alias("articles_count"),
            F.sum(F.when(F.length(F.trim(F.col("text_stats"))) == 0, 1).otherwise(0)).alias("empty_text"),
            F.min("text_len").alias("min"),
            F.expr("percentile_approx(text_len, 0.5)").alias("p50"),
            F.expr("percentile_approx(text_len, 0.9)").alias("p90"),
            F.expr("percentile_approx(text_len, 0.99)").alias("p99"),
            F.max("text_len").alias("max"),
            F.avg("text_len").alias("avg"),
            F.stddev("text_len").alias("stddev"),
        )
        .first()
    )

    articles_count = _as_int(articles_len_row["articles_count"])
    empty_text_articles = _as_int(articles_len_row["empty_text"])
    print(f">>> articles={articles_count:,} ({_safe_ratio(articles_count, total):.2%})")
    print(f">>> empty_text_articles={empty_text_articles:,} ({_safe_ratio(empty_text_articles, articles_count):.2%})")
    spark = df.sparkSession
    global_len_df = spark.createDataFrame(
        [
            {
                "scope": "global",
                "empty_text": int(empty_text),
                "min": _as_int(global_len_row["min"]),
                "p50": _as_int(global_len_row["p50"]),
                "p90": _as_int(global_len_row["p90"]),
                "p99": _as_int(global_len_row["p99"]),
                "max": _as_int(global_len_row["max"]),
                "avg": _as_float(global_len_row["avg"]),
                "stddev": _as_float(global_len_row["stddev"]),
            }
        ]
    )
    articles_len_df = spark.createDataFrame(
        [
            {
                "scope": "articles",
                "articles_count": int(articles_count),
                "empty_text": int(empty_text_articles),
                "min": _as_int(articles_len_row["min"]),
                "p50": _as_int(articles_len_row["p50"]),
                "p90": _as_int(articles_len_row["p90"]),
                "p99": _as_int(articles_len_row["p99"]),
                "max": _as_int(articles_len_row["max"]),
                "avg": _as_float(articles_len_row["avg"]),
                "stddev": _as_float(articles_len_row["stddev"]),
            }
        ]
    )

    print(">>> text length summary (global):")
    global_len_df.show(truncate=False)

    print(">>> text length summary (articles):")
    articles_len_df.show(truncate=False)


    # 5) Link density + noise indicators
    # Prefer extracted arrays from the parser (links/categories/templates).
    # Fall back to counting markup in the rawest available wikitext.
    def _count_array_or_markup(df_in, array_col: str, markup_pattern: str):
        if array_col in df_in.columns:
            return F.coalesce(F.size(F.col(array_col)), F.lit(0))

        # For markup-based counts, prefer the rawest available text.
        raw_text_col = pick_text_col(
            df_in,
            candidates=["text_raw", "text", "revision.text._VALUE", "revision.text"],
        )
        return split_count(
            F.coalesce(raw_text_col.cast("string"), F.lit("")),
            markup_pattern,
        )

    link_count_col = _count_array_or_markup(df_articles, "links", r"\[\[")
    category_count_col = _count_array_or_markup(df_articles, "categories", r"\[\[Category:")
    template_count_col = _count_array_or_markup(df_articles, "templates", r"\{\{")

    df_links = (
        df_articles.withColumn("link_count", link_count_col)
        .withColumn("category_count", category_count_col)
        .withColumn("template_count", template_count_col)
        .withColumn(
            "links_per_kchar",
            F.when(
                F.col("text_len") > 0,
                F.col("link_count") / (F.col("text_len") / F.lit(1000.0)),
            ).otherwise(F.lit(0.0)),
        )
    )

    link_summary_df = df_links.select(
        F.min("link_count").alias("min_links"),
        F.expr("percentile_approx(link_count, 0.5)").alias("p50_links"),
        F.expr("percentile_approx(link_count, 0.9)").alias("p90_links"),
        F.max("link_count").alias("max_links"),
        F.avg("links_per_kchar").alias("avg_links_per_kchar"),
        F.expr("percentile_approx(links_per_kchar, 0.9)").alias("p90_links_per_kchar"),
        F.avg("category_count").alias("avg_categories"),
        F.avg("template_count").alias("avg_templates"),
    )
    top_most_linked_df = None
    if has_title:
        top_most_linked_df = (
            df_links.orderBy(F.desc("link_count"))
            .select(
                "title",
                "text_len",
                "link_count",
                "links_per_kchar",
                "category_count",
                "template_count",
            )
            .limit(200)
        )

    print(">>> link density summary (articles):")
    link_summary_df.show(truncate=False)

    if top_most_linked_df is not None:
        print(">>> most-linked pages (top 25):")
        top_most_linked_df.show(25, truncate=False)


    # 6) Vocabulary (simple)
    # IMPORTANT: tokenizer input must exist in the dataset passed to transform.
    # We created "text_stats", so we must select that column (not "text").
    tok = RegexTokenizer(
        inputCol="text_stats",
        outputCol="tokens_raw",
        pattern="\\W+",
        toLowercase=True,
        minTokenLength=3,
    )

    # Keep only the needed column to reduce shuffle
    df_tok = tok.transform(df_links.select("text_stats"))

    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
    df_tok2 = remover.transform(df_tok)

    top_tokens_df = (
        df_tok2.select(F.explode("tokens").alias("token"))
        .groupBy("token")
        .count()
        .orderBy(F.desc("count"))
        .limit(int(top_k))
    )

    print(">>> top tokens (top 25):")
    top_tokens_df.show(25, truncate=False)

    overview_df = spark.createDataFrame(
        [
            {
                "generated_at": generated_at,
                "out_dir": str(out_dir),
                "total_pages": int(total),
                "redirect_pages": int(redirects_global),
                "distinct_titles": int(distinct_titles),
                "missing_titles": int(missing_titles),
                "duplicate_title_groups": int(dup_titles),
                "articles_count": int(articles_count),
                "empty_text_global": int(empty_text),
                "empty_text_articles": int(empty_text_articles),
                "columns": list(df.columns),
            }
        ]
    )
    
    _write_stats_outputs(
        out_dir=out_path,
        overview_df=overview_df,
        ns_dist_df=ns_dist_df.orderBy(F.desc("count")) if ns_dist_df is not None else None,
        global_len_df=global_len_df,
        articles_len_df=articles_len_df,
        link_summary_df=link_summary_df,
        top_most_linked_df=top_most_linked_df,
        top_tokens_df=top_tokens_df,
    )

    print(f"\n>>> Saved stats artifacts to: {out_path}")
