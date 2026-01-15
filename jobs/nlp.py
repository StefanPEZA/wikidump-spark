from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
    IDF,
    Normalizer,
    SQLTransformer,
)
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import LDA


def prepare_docs_for_nlp(df, min_chars_src: int = 500, min_chars_clean: int = 300):
    """Prepare a wiki pages DataFrame for NLP.

    This function produces a DataFrame with columns:
            - id, title, text

    Notes:
        - The fitted Spark ML model saved by `run_nlp` expects a `text` column.
    """

    if df is None:
        raise ValueError("df is None")

    if "text" not in df.columns:
        raise ValueError(
            f"Missing required column 'text'. Ensure parser.py is run before NLP. Got: {df.columns}"
        )

    docs = (
        df.filter((F.col("ns") == 0) & F.col("redirect").isNull())
        .select("id", "title", F.col("text").cast("string").alias("text"))
        .withColumn("text", F.coalesce(F.col("text"), F.lit("")))
        .filter(F.length(F.col("text")) > int(min_chars_src))
        .filter(F.length(F.col("text")) > int(min_chars_clean))
    )

    return docs


def _cluster_label_from_terms(terms: list[str]) -> str:
    """Create a short, human-readable label from top terms."""
    clean = [t for t in terms if t and isinstance(t, str)]
    clean = [t.replace("_", " ") for t in clean]
    clean = [t.strip() for t in clean if t.strip()]
    # Use first 3 terms as a compact label
    head = clean[:3]
    return " / ".join(head) if head else "(unlabeled)"


def _argmax_from_vector_col(vec_col: str) -> F.Column:
    """Return the argmax index for a Spark ML vector column."""
    arr = vector_to_array(F.col(vec_col))
    # argmax index = position(max_value) - 1 (array_position is 1-based)
    return F.array_position(arr, F.array_max(arr)) - F.lit(1)


def top_topics_per_article(
    out_topics: DataFrame,
    topics_df: DataFrame,
    n: int = 3,
) -> DataFrame:
    """Return (id, title, topic, topic_name, topic_weight, topic_rank) for top-N topics per article."""
    arr = vector_to_array(F.col("topic_distribution"))
    exploded = out_topics.select(
        "id",
        "title",
        F.posexplode(arr).alias("topic", "topic_weight"),
    )

    w = Window.partitionBy("id").orderBy(F.desc("topic_weight"), F.asc("topic"))
    ranked = exploded.withColumn("topic_rank", F.row_number().over(w)).filter(
        F.col("topic_rank") <= F.lit(int(n))
    )

    return (
        ranked.join(topics_df.select("topic", "topic_name"), on="topic", how="left")
        .select(
            F.col("id").cast("long").alias("id"),
            F.col("title").cast("string").alias("title"),
            F.col("topic").cast("int").alias("topic"),
            F.col("topic_name").cast("string").alias("topic_name"),
            F.col("topic_weight").cast("double").alias("topic_weight"),
            F.col("topic_rank").cast("int").alias("topic_rank"),
        )
    )


def build_stopwords() -> list[str]:
    base_sw = StopWordsRemover.loadDefaultStopWords("english")
    extra_sw = [
        "align", "style", "rowspan", "colspan", "nbsp", "bgcolor", "scope", "width", "height", "valign", "class", "px",
        "left", "right", "center", "reflist", "cite", "references", "defaultsort", "image", "thumb", "file", "category",
        "www", "http", "https", "com", "org", "html", "htm", "url", "born", "died", "list", "refer", "stub",
        "alignbars", "gridcolor", "lightgrey", "background", "color", "solid", "text", "near", "start", "end", "till", "bar",
        "also", "may", "used", "new", "one", "two", "first",
        "american", "state", "states", "united", "population", "census", "time", "year", "years",
        "world", "total", "swiss", "apartments", "living", "area", "family", "film", "school", "national", "university",
    ]

    # unique + lowercase
    return sorted({w.lower() for w in (list(base_sw) + extra_sw) if w and isinstance(w, str)})


def build_text_processing_stages(stop_words: list[str]):
    """Create reusable text->tokens->terms stages.

    Output columns:
      - tokens_raw
      - tokens
      - bigrams
      - terms
    """
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens_raw",
        pattern="[^a-z]+",
        toLowercase=True,
        minTokenLength=3,
    )

    remover = StopWordsRemover(
        inputCol="tokens_raw",
        outputCol="tokens",
        stopWords=stop_words,
    )

    ngram = NGram(n=2, inputCol="tokens", outputCol="bigrams")
    merge_terms = SQLTransformer(statement="SELECT *, concat(tokens, bigrams) AS terms FROM __THIS__")

    return tokenizer, remover, ngram, merge_terms


def build_count_vectorizer(
    input_col: str = "terms",
    output_col: str = "tf",
    vocab_size: int = 50000,
    min_df: int = 20,
    max_df: float = 0.6,
) -> CountVectorizer:
    return CountVectorizer(
        inputCol=input_col,
        outputCol=output_col,
        vocabSize=vocab_size,
        minDF=min_df,
        maxDF=max_df,
    )


def train_bkmeans_clustering(docs: DataFrame, k: int) -> tuple[PipelineModel, DataFrame]:
    """Train TF-IDF + BisectingKMeans clustering model and return (model, transformed_df)."""
    wiki_sw = build_stopwords()
    tokenizer, remover, ngram, merge_terms = build_text_processing_stages(wiki_sw)

    cv = build_count_vectorizer()
    idf = IDF(inputCol="tf", outputCol="tfidf", minDocFreq=10)
    norm = Normalizer(inputCol="tfidf", outputCol="features", p=2.0)

    bkmeans = BisectingKMeans(
        featuresCol="features",
        predictionCol="cluster",
        k=k,
        seed=42,
        minDivisibleClusterSize=1.0,
        maxIter=40,
    )

    cluster_pipe = Pipeline(stages=[tokenizer, remover, ngram, merge_terms, cv, idf, norm, bkmeans])
    model = cluster_pipe.fit(docs)
    out_df = model.transform(docs)
    return model, out_df


def build_cluster_labels(cluster_model: PipelineModel, spark_session) -> DataFrame:
    """Create a DataFrame of cluster labels + top terms."""
    vocab = cluster_model.stages[4].vocabulary
    centers = cluster_model.stages[-1].clusterCenters()

    label_rows = []
    for i, c in enumerate(centers):
        arr = list(c)
        top_idx = sorted(range(len(arr)), key=lambda j: arr[j], reverse=True)[:15]
        terms = [vocab[j] for j in top_idx]
        name = _cluster_label_from_terms(terms)
        label_rows.append({"cluster": int(i), "cluster_name": name, "top_terms": terms})

    return spark_session.createDataFrame(label_rows)


def save_clustering_outputs(
    out_cluster: DataFrame,
    cluster_model: PipelineModel,
    out_base: str,
    labels_df: DataFrame,
) -> None:
    (out_cluster.select("id", "title", "cluster")
        .write.mode("overwrite")
        .parquet(f"{out_base}/clusters/assignments"))

    cluster_model_path = f"{out_base}/clusters/model"
    cluster_model.write().overwrite().save(cluster_model_path)
    print(f">>> Saved clustering model: {cluster_model_path}")

    labels_df.write.mode("overwrite").parquet(f"{out_base}/clusters/labels")

    (out_cluster.select("id", "title", "cluster")
        .join(labels_df.select("cluster", "cluster_name"), on="cluster", how="left")
        .write.mode("overwrite")
        .parquet(f"{out_base}/clusters/assignments_labeled"))


def train_lda_topics(docs: DataFrame, k: int) -> tuple[PipelineModel, DataFrame, DataFrame]:
    """Train LDA topic model and return (model, transformed_df, labels_df)."""
    wiki_sw = build_stopwords()
    tokenizer, remover, ngram, merge_terms = build_text_processing_stages(wiki_sw)

    cv = build_count_vectorizer()

    lda = LDA(
        k=k,
        maxIter=40,
        featuresCol="tf",
        seed=42,
        optimizer="online",
        topicDistributionCol="topic_distribution",
    )

    topic_pipe = Pipeline(stages=[tokenizer, remover, ngram, merge_terms, cv, lda])
    model = topic_pipe.fit(docs)
    out_topics = model.transform(docs)

    spark = out_topics.sparkSession
    cv_model = model.stages[4]
    lda_model = model.stages[-1]
    topic_vocab = cv_model.vocabulary

    topic_rows = []
    for row in lda_model.describeTopics(maxTermsPerTopic=15).collect():
        idxs = row["termIndices"] or []
        weights = row["termWeights"] or []
        terms = [topic_vocab[i] for i in idxs]
        name = _cluster_label_from_terms(terms)
        topic_rows.append(
            {
                "topic": int(row["topic"]),
                "topic_name": name,
                "top_terms": terms,
                "term_weights": [float(w) for w in weights],
            }
        )

    labels_df = spark.createDataFrame(topic_rows)
    return model, out_topics, labels_df


def save_topic_outputs(
    out_topics: DataFrame,
    topic_model: PipelineModel,
    out_base: str,
    topics_df: DataFrame,
) -> None:
    topic_model_path = f"{out_base}/topics/model"
    topic_model.write().overwrite().save(topic_model_path)
    print(f">>> Saved topic model: {topic_model_path}")

    topics_df.write.mode("overwrite").parquet(f"{out_base}/topics/labels")

    out_topics = out_topics.withColumn("topic", _argmax_from_vector_col("topic_distribution").cast("int"))

    (out_topics.select("id", "title", "topic", "topic_distribution")
        .write.mode("overwrite")
        .parquet(f"{out_base}/topics/assignments"))

    (out_topics.select("id", "title", "topic")
        .join(topics_df.select("topic", "topic_name"), on="topic", how="left")
        .write.mode("overwrite")
        .parquet(f"{out_base}/topics/assignments_labeled"))


def load_clustering_model(model_path: str) -> PipelineModel:
    """Load the saved clustering PipelineModel from disk."""
    return PipelineModel.load(model_path)


def load_topic_model(model_path: str) -> PipelineModel:
    """Load the saved topic (LDA) PipelineModel from disk."""
    return PipelineModel.load(model_path)


def predict_clusters(df: DataFrame, model_path: str, labels_path: str | None = None) -> DataFrame:
    """Assign clusters to new data using a previously saved clustering model.

    Requirements:
    - Input df must contain the parser-cleaned column `text`.
    """
    docs = prepare_docs_for_nlp(df)
    model = load_clustering_model(model_path)
    out = model.transform(docs).select("id", "title", F.col("cluster").cast("int").alias("cluster"))

    if labels_path:
        labels = df.sparkSession.read.parquet(labels_path).select("cluster", "cluster_name")
        out = out.join(labels, on="cluster", how="left")

    return out


def predict_topics(df: DataFrame, model_path: str, labels_path: str | None = None) -> DataFrame:
    """Assign dominant topics to new data using a previously saved LDA topic model.

    Requirements:
    - Input df must contain the parser-cleaned column `text`.
    """
    docs = prepare_docs_for_nlp(df)
    model = load_topic_model(model_path)
    out = model.transform(docs)
    out = out.withColumn("topic", _argmax_from_vector_col("topic_distribution").cast("int"))
    out = out.select("id", "title", "topic", "topic_distribution")

    if labels_path:
        labels = df.sparkSession.read.parquet(labels_path).select("topic", "topic_name")
        out = out.join(labels, on="topic", how="left")

    return out


def run_nlp(df, out_base: str, k: int = 10):
    print(">>> [3/4] NLP: Topics (LDA) + Clustering (TF-IDF + BisectingKMeans)...")

    if df is None:
        print(">>> ERROR: DataFrame is None")
        return

    if not out_base:
        print(">>> ERROR: out_base is required (pass args.out from main.py).")
        return

    try:
        docs = prepare_docs_for_nlp(df)
    except Exception as e:
        print(f">>> ERROR: Failed to prepare docs for NLP: {e}")
        return

    print(">>> NLP sanity: showing cleaned text head")
    docs.select("title", F.substring("text", 1, 240).alias("text_head")).show(3, truncate=False)

    # 1) Clustering (BKMeans)
    cluster_model, out_cluster = train_bkmeans_clustering(docs, k=k)

    spark = out_cluster.sparkSession
    labels_df = build_cluster_labels(cluster_model, spark)

    print(">>> Cluster sizes (with labels):")
    (
        out_cluster.groupBy("cluster").count()
        .join(labels_df.select("cluster", "cluster_name"), on="cluster", how="left")
        .orderBy(F.desc("count"))
        .select("cluster", "cluster_name", "count")
        .show(25, truncate=False)
    )

    print(">>> Top terms per cluster:")
    for r in labels_df.orderBy("cluster").collect():
        print(
            f" - Cluster {r['cluster']} ({r['cluster_name']}): {', '.join((r['top_terms'] or [])[:12])}"
        )

    save_clustering_outputs(out_cluster, cluster_model, out_base, labels_df)

    # 2) Topic discovery (LDA) on term counts
    topic_model, out_topics, topics_df = train_lda_topics(docs, k=k)
    
    print(">>> Topic prevalence (dominant topic per doc):")
    out_topics_with_topic = out_topics.withColumn(
        "topic", _argmax_from_vector_col("topic_distribution").cast("int")
    )
    (
        out_topics_with_topic.groupBy("topic").count()
        .join(topics_df.select("topic", "topic_name"), on="topic", how="left")
        .orderBy(F.desc("count"))
        .select("topic", "topic_name", "count")
        .show(25, truncate=False)
    )

    print(">>> Top terms per topic:")
    for r in topics_df.orderBy("topic").collect():
        print(
            f" - Topic {r['topic']} ({r['topic_name']}): {', '.join((r['top_terms'] or [])[:12])}"
        )

    save_topic_outputs(out_topics, topic_model, out_base, topics_df)

    try:
        docs.unpersist()
    except Exception:
        pass
