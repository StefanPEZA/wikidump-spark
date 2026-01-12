from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer
)
from pyspark.ml.clustering import BisectingKMeans


def run_nlp(df, out_base: str, k: int = 10):
    print(">>> [3/4] NLP: Clustering (TF-IDF + KMeans)...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    if "text" in df.columns:
        src_col = "text"
    elif "text_raw" in df.columns:
        src_col = "text_raw"
    else:
        print(f">>> ERROR: Missing text column. Expected 'text' or 'text_raw'. Got: {df.columns}")
        return

    docs = (
        df.filter((F.col("ns") == 0) & F.col("redirect").isNull())
          .select("id", "title", F.col(src_col).alias("text_src"))
          .withColumn("text_nlp", F.coalesce(F.col("text_src").cast("string"), F.lit("")))
    )

    docs = docs.filter(F.length(F.col("text_nlp")) > 500)

    docs = (
        docs
        # Remove tables/templates leftovers (defensive)
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"(?is)\{\|.*?\|\}", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"(?is)\{\{.*?\}\}", " "))

        # Remove refs/html
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"(?is)<ref[^>]*>.*?</ref>", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"(?i)<ref[^>]*/>", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"(?is)<(gallery|script|style|noinclude)[^>]*>.*?</\1>", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"<[^>]+>", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"&[a-zA-Z]+;", " "))

        # Remove Category/File/Image wiki links
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\[\[(?i:(Category|File|Image)):[^\]]*?\]\]", " "))

        # Flatten wiki links
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\[\[[^|\]]+\|([^\]]+)\]\]", r"$1"))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\[\[([^\]]+)\]\]", r"$1"))

        # External links
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"$1"))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\[https?://[^\]]+\]", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"https?://\S+", " "))

        # Kill common html/wiki noise words
        .withColumn(
            "text_nlp",
            F.regexp_replace(
                "text_nlp",
                r"(?i)\b(align|style|rowspan|colspan|nbsp|bgcolor|scope|width|height|valign|class|px|left|right|center)\b",
                " "
            )
        )
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\b[a-f]{6}\b", " "))

        # Keep letters/spaces only
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"[^a-zA-Z\s]", " "))
        .withColumn("text_nlp", F.regexp_replace("text_nlp", r"\s+", " "))
        .withColumn("text_nlp", F.trim("text_nlp"))
        .filter(F.length("text_nlp") > 300)
    )

    # Filter pages that are mostly table markup (many pipes)
    pipe_count = (F.length("text_src") - F.length(F.regexp_replace("text_src", r"\|", "")))
    docs = docs.filter(pipe_count < 250)

    print(f">>> NLP sanity: using '{src_col}' -> text_nlp")
    docs.select("title", F.substring("text_nlp", 1, 240).alias("text_head")).show(3, truncate=False)

    base_sw = StopWordsRemover.loadDefaultStopWords("english")
    extra_sw = [
        "align", "style", "rowspan", "colspan", "nbsp", "bgcolor", "scope", "width", "height", "valign", "class", "px",
        "left", "right", "center", "reflist", "cite", "references", "defaultsort", "image", "thumb", "file", "category",
        "www", "http", "https", "com", "org", "html", "htm", "url", "born", "died", "list", "refer", "stub",
        "alignbars", "gridcolor", "lightgrey", "background", "color", "solid", "text", "near", "start", "end", "till", "bar",
        "also", "may", "used", "new", "one", "two", "first",
        "american", "state", "states", "united", "population", "census", "born", "died", "time", "year", "years",
        "world", "total", "swiss", "apartments", "living", "area", "family", "film", "school", "national", "university"
    ]

    # unique + lowercase
    wiki_sw = sorted({w.lower() for w in (list(base_sw) + extra_sw) if w and isinstance(w, str)})

    tokenizer = RegexTokenizer(
        inputCol="text_nlp",
        outputCol="tokens_raw",
        pattern="[^a-z]+",
        toLowercase=True,
        minTokenLength=3
    )

    remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        stopWords=wiki_sw
    )

    cv = CountVectorizer(
        inputCol="tokens",
        outputCol="tf",
        vocabSize=30000,
        minDF=50,
        maxDF=0.8
    )

    idf = IDF(
        inputCol="tf",
        outputCol="tfidf",
        minDocFreq=10
    )

    norm = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

    bkmeans = BisectingKMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        seed=42,
        minDivisibleClusterSize=1.0,
        maxIter=40
    )

    pipe = Pipeline(stages=[tokenizer, remover, cv, idf, norm, bkmeans])
    model = pipe.fit(docs)
    out = model.transform(docs)

    print(">>> Cluster sizes:")
    out.groupBy("cluster").count().orderBy(F.desc("count")).show(50, truncate=False)

    (out.select("id", "title", "cluster")
        .write.mode("overwrite")
        .parquet(f"{out_base}/clusters/assignments"))

    vocab = model.stages[2].vocabulary
    centers = model.stages[-1].clusterCenters()

    print(">>> Top terms per cluster:")
    for i, c in enumerate(centers):
        arr = list(c)
        top_idx = sorted(range(len(arr)), key=lambda j: arr[j], reverse=True)[:15]
        terms = [vocab[j] for j in top_idx]
        print(f" - Cluster {i}: {', '.join(terms)}")
