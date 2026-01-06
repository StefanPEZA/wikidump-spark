from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

from jobs.utils.wiki_df_helpers import (
    pick_text_col,
    split_count,
    print_len_summary,
)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def run_stats(df):
    print(">>> [2/4] EDA: Statistics...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    has_title = "title" in df.columns
    has_ns = "ns" in df.columns
    has_redirect = "redirect" in df.columns

    # 1) Basic counts (global)
    total = df.count()
    print(f"\n>>> Total pages (rows): {total:,}")

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
        dup_titles = df.groupBy("title").count().filter(F.col("count") > 1).count()
        print(f">>> Distinct titles: {title_stats['distinct_titles']:,}")
        print(f">>> Missing/empty titles: {title_stats['missing_titles']:,}")
        print(f">>> Duplicate title groups: {dup_titles:,}")

    if has_ns:
        print("\n>>> Namespace distribution (top 20):")
        df.groupBy("ns").count().orderBy(F.desc("count")).show(20, truncate=False)

    if has_redirect:
        redirects_global = df.filter(F.col("redirect").isNotNull()).count()
        print(
            f"\n>>> Redirect pages (all namespaces): {redirects_global:,} ({_safe_ratio(redirects_global, total):.2%})"
        )

    # 2) Choose text column
    # Prefer cleaned parser output, but keep backward compatibility with raw XML variants.
    text_col = pick_text_col(df)

    if text_col is None:
        print(">>> Text column not found. Skipping text-based stats.")
        return

    # normalize chosen text to a string column "text" (clean when available)
    df_base = df.withColumn("text", text_col.cast("string"))

    # 3) Global text stats
    df_global = df_base.withColumn(
        "text_len", F.length(F.coalesce(F.col("text"), F.lit("")))
    )
    empty_text = df_global.filter(
        F.col("text").isNull() | (F.length(F.trim(F.col("text"))) == 0)
    ).count()
    print(
        f"\n>>> Empty/missing text (global): {empty_text:,} ({_safe_ratio(empty_text, total):.2%})"
    )
    print_len_summary(df_global, "global")

    # 4) Focused dataset: articles (ns=0) and non-redirect
    df_articles = df_global
    if has_ns:
        df_articles = df_articles.filter(F.col("ns").cast("int") == 0)
    if has_redirect:
        df_articles = df_articles.filter(F.col("redirect").isNull())

    articles_count = df_articles.count()
    print(
        f"\n>>> Articles only (ns=0, non-redirect): {articles_count:,} ({_safe_ratio(articles_count, total):.2%})"
    )
    print_len_summary(df_articles, "articles(ns=0, non-redirect)")

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

    print("\n>>> Link density summary (articles):")
    (df_links.select(
        F.min("link_count").alias("min_links"),
        F.expr("percentile_approx(link_count, 0.5)").alias("p50_links"),
        F.expr("percentile_approx(link_count, 0.9)").alias("p90_links"),
        F.max("link_count").alias("max_links"),
        F.avg("links_per_kchar").alias("avg_links_per_kchar"),
        F.expr("percentile_approx(links_per_kchar, 0.9)").alias("p90_links_per_kchar"),
    ).show(truncate=False))

    if has_title:
        print(">>> Top 20 most-linked pages (articles):")
        (df_links.orderBy(F.desc("link_count"))
            .select("title", "text_len", "link_count", "links_per_kchar", "category_count", "template_count")
            .show(20, truncate=False))

    # 6) Vocabulary (simple)
    print("\n>>> Vocabulary (articles): top 30 tokens (stopwords removed):")
    tok = RegexTokenizer(
        inputCol="text",
        outputCol="tokens_raw",
        pattern="\\W+",
        toLowercase=True,
        minTokenLength=3
    )
    df_tok = tok.transform(df_links.select("text"))

    remover = StopWordsRemover(inputCol="tokens_raw", outputCol="tokens")
    df_tok2 = remover.transform(df_tok)

    (df_tok2.select(F.explode("tokens").alias("token"))
        .groupBy("token").count()
        .orderBy(F.desc("count"))
        .show(30, truncate=False))
