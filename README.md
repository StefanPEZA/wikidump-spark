# wikidump-spark

## How Each Log Section Is Computed

### [1/4] ETL: Parsing & Cleaning
Source: `jobs/parser.py` (`run_parser`, `_clean_wikitext`)

- **XML load**: `spark.read.format("xml").option("rowTag", "page").load(input_path)`
- **Column selection/casting**: `select(...)` with `cast(...)`, `trim(...)` and helper `_pick_col(...)` / `pick_text_col(...)`
- **Regex extraction (arrays)** from `text_raw`:
	- Categories: `regexp_extract_all(text_raw, r"\[\[(?i:Category):([^\]\|]+)")`
	- Templates: `regexp_extract_all(text_raw, r"\{\{\s*([^\|\}\n<]+)")`
	- External links: `regexp_extract_all(text_raw, r"(https?://[^\]\|\n<>]+)")`
	- Internal links: `regexp_extract_all(text_raw, r"\[\[([^\]\|#]+)")` + `regexp_replace(^:+, "")`
- **Array cleanup**: `transform(...)` + `trim(...)` + `filter(...)` + `array_distinct(...)` (drops empty/duplicate tokens)
- **Text cleaning**: `_clean_wikitext(...)` uses a sequence of `regexp_replace(...)` operations to remove refs/html/templates/tables and flatten wiki links
- **Sample check output**: `where(...)`, `limit(1)`, `collect()` then prints fields from the collected Row

### [2/4] EDA: Statistics
Source: `jobs/stats.py` (`run_stats`)

- **Counts**:
	- Total pages / redirects: `df.agg(count(1), sum(when(redirect is not null, 1)))`
	- Title stats: `countDistinct(title)`, `sum(when(title is null/blank, 1))`, duplicates via `groupBy(title).count().filter(count>1).count()`
- **Namespace distribution table**: `groupBy("ns").count().orderBy(desc("count")).show(...)`
- **Text-length summaries**:
	- Build `text_stats` (chosen via `pick_text_col`) and `text_len = length(text_stats)`
	- Percentiles: `expr("percentile_approx(text_len, p)")`
	- Other stats: `min/max/avg/stddev`, empty text via `sum(when(length(trim(text))==0,1))`
- **Link density summary** (articles):
	- Link/category/template counts from arrays using helper `split_count(...)` (then aggregations and percentiles)
- **Top tokens**:
	- `RegexTokenizer` + `StopWordsRemover`, then `explode(tokens)` + `groupBy(token).count().orderBy(desc(count)).show(...)`

### [3/4] NLP: Topics (LDA) + Clustering (TF-IDF + BisectingKMeans)
Source: `jobs/nlp.py` (`run_nlp`)

- **Doc filtering**: `filter(ns==0 & redirect is null)` + length filters, then `select(id,title,text)`
- **Text head preview**: `substring(text, 1, 240)` + `show(...)`

**Clustering (BisectingKMeans) pipeline**
- Tokenization: `RegexTokenizer` (splits on non-letters)
- Stopwords: `StopWordsRemover` (custom + default English)
- Bigrams: `NGram(n=2)`
- Terms merge: `SQLTransformer("concat(tokens, bigrams) AS terms")`
- Vectorization: `CountVectorizer` -> `IDF` -> `Normalizer(p=2)`
- Clustering: `BisectingKMeans(k=K, seed=42, maxIter=40)`
- **Cluster sizes**: `groupBy("cluster").count()` then `join(labels_df)` and `orderBy(desc(count)).show(...)`
- **Top terms per cluster**: use `CountVectorizerModel.vocabulary` + `BisectingKMeansModel.clusterCenters()` and take top weights

**Topic modeling (LDA) pipeline**
- Same preprocessing through `CountVectorizer` (term counts)
- Topics: `LDA(k=K, optimizer="online", maxIter=40, topicDistributionCol="topic_distribution")`
- **Topic prevalence**:
	- Dominant topic = `argmax(topic_distribution)` implemented via `vector_to_array(...)`, `array_max(...)`, `array_position(...)`
	- Then `groupBy("topic").count().join(topics_df).show(...)`
- **Top terms per topic**: `LDAModel.describeTopics(maxTermsPerTopic=15)` + map `termIndices` -> vocabulary

### [4/4] Graph: Building Knowledge Graph (+ PageRank)
Source: `jobs/graph.py` (`run_graph`)

- **Vertices (pages)**: `select(id,title)` + `dropna` + `dropDuplicates` + normalization (`regexp_replace`, `lower`)
- **Edges (page->page links)**:
	- `explode_outer(links)`
	- Normalize link titles
	- Join to destination pages on normalized title
	- Aggregate weights: `groupBy(src_id, dst_id).count()`
- **PageRank (GraphFrames)**:
	- Build `GraphFrame(vertices, edges)`
	- Run `pageRank(resetProbability=0.15, maxIter=N)`
	- Print top nodes: `orderBy(desc(pagerank)).show(...)`