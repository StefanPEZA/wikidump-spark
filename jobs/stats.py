from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

from jobs.utils.wiki_df_helpers import (
    pick_text_col,
    split_count,
    print_len_summary,
)


def run_stats(df):
    print(">>> [2/4] EDA: Statistics...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    # 0) Schema
    print(">>> Schema:")
    df.printSchema()

    # 1) Basic counts (global)
    total = df.count()
    print(f"\n>>> Total pages (rows): {total:,}")

    if "title" in df.columns:
        distinct_titles = df.select("title").distinct().count()
        title_missing = df.filter(F.col("title").isNull() | (F.trim(F.col("title")) == "")).count()
        dup_titles = df.groupBy("title").count().filter(F.col("count") > 1).count()
        print(f">>> Distinct titles: {distinct_titles:,}")
        print(f">>> Missing/empty titles: {title_missing:,}")
        print(f">>> Duplicate title groups: {dup_titles:,}")

    if "ns" in df.columns:
        print("\n>>> Namespace distribution (top 20):")
        df.groupBy("ns").count().orderBy(F.desc("count")).show(20, truncate=False)

    if "redirect" in df.columns:
        redirects_global = df.filter(F.col("redirect").isNotNull()).count()
        print(f"\n>>> Redirect pages (all namespaces): {redirects_global:,} ({redirects_global/total:.2%})")

    # 2) Choose text column
    text_col = pick_text_col(df)
    if text_col is None:
        print(">>> Text column not found. Skipping text-based stats.")
        return

    # normalize text to a string column "text"
    df_base = df.withColumn("text", text_col.cast("string"))

    # 3) Global text stats
    df_global = df_base.withColumn("text_len", F.length(F.col("text")))
    empty_text = df_global.filter(
        F.col("text").isNull() | (F.length(F.trim(F.col("text"))) == 0)
    ).count()
    print(f"\n>>> Empty/missing text (global): {empty_text:,} ({empty_text/total:.2%})")
    print_len_summary(df_global, "global")

    # 4) Focused dataset: articles (ns=0) and non-redirect
    df_articles = df_global
    if "ns" in df.columns:
        df_articles = df_articles.filter(F.col("ns").cast("int") == 0)
    if "redirect" in df.columns:
        df_articles = df_articles.filter(F.col("redirect").isNull())

    articles_count = df_articles.count()
    print(f"\n>>> Articles only (ns=0, non-redirect): {articles_count:,} ({articles_count/total:.2%})")
    print_len_summary(df_articles, "articles(ns=0, non-redirect)")

    # 5) Link density + noise indicators
    df_links = (df_articles
        .withColumn("link_count", split_count(F.col("text"), r"\[\["))
        .withColumn("category_count", split_count(F.col("text"), r"\[\[Category:"))
        .withColumn("template_count", split_count(F.col("text"), r"\{\{"))
        .withColumn("links_per_kchar", F.col("link_count") / (F.col("text_len") / 1000.0))
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

    if "title" in df.columns:
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
