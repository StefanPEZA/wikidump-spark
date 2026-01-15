from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from jobs.utils.wiki_df_helpers import _pick_col, pick_text_col

def _clean_wikitext(text_raw: F.Column) -> F.Column:
    text = F.coalesce(text_raw.cast("string"), F.lit(""))

    # --- Remove references ---
    text = F.regexp_replace(text, r"(?is)<ref[^>]*>.*?</ref>", " ")
    text = F.regexp_replace(text, r"(?i)<ref[^>]*/>", " ")

    # --- Remove non-readable blocks ---
    text = F.regexp_replace(text, r"(?is)<(gallery|script|style|noinclude)[^>]*>.*?</$1>", " ")
    text = F.regexp_replace(text, r"(?i)<br\s*/?>", "\n")
    text = F.regexp_replace(text, r"<[^>]+>", " ")

    # --- Remove Category, File, Image links ---
    text = F.regexp_replace(text, r"\[\[(?i:(Category|File|Image)):[^\]]*?\]\]", " ")

    # --- Flatten templates ---
    # Remove Infobox params (lines starting with |)
    text = F.regexp_replace(text, r"(?m)^\s*\|.*$", "")
    # Generic templates: {{...}} -> remove entirely. 
    # Repeat to handle nesting (e.g. {{Infobox... {{date}} ...}}) as regex is not recursive.
    # Level 1 (innermost)
    text = F.regexp_replace(text, r"\{\{[^\{\}]*\}\}", "")
    # Level 2
    text = F.regexp_replace(text, r"\{\{[^\{\}]*\}\}", "")
    # Level 3 (cleanup remaining)
    text = F.regexp_replace(text, r"(?s)\{\{[^}]*?\}\}", "") 

    # --- Flatten internal links ---
    # [[target|label]] -> label
    text = F.regexp_replace(text, r"\[\[[^|\]]+\|([^\]]+)\]\]", r"$1")
    # [[target]] -> target
    text = F.regexp_replace(text, r"\[\[([^\]]+)\]\]", r"$1")

    # --- Flatten external links ---
    # [http://url label] -> label
    text = F.regexp_replace(text, r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"$1")
    # [http://url] -> remove
    text = F.regexp_replace(text, r"\[https?://[^\]]+\]", " ")
    # Bare URLs -> remove
    text = F.regexp_replace(text, r"https?://\S+", " ")

    # --- Table & structure cleanup ---
    text = F.regexp_replace(text, r"(?m)^\s*\{\|.*$", "")
    text = F.regexp_replace(text, r"(?m)^\s*\|\}.*$", "")
    text = F.regexp_replace(text, r"(?m)^\s*\|-.*$", "")
    text = F.regexp_replace(text, r"(?m)^\s*\|\s*", "")

    # --- Headers ---
    text = F.regexp_replace(text, r"(?m)^=+\s*(.*?)\s*=+$", r"$1")

    # --- Formatting cleanup ---
    text = F.regexp_replace(text, r"'{2,5}", "")              # bold/italic
    text = F.regexp_replace(text, r"(?m)^\s*[*#;:]+\s*", "")  # bullets
    text = F.regexp_replace(text, r"[ \t]+", " ")             # collapse spaces
    text = F.regexp_replace(text, r"(?m)^[ \t]+|[ \t]+$", "") # trim lines
    text = F.regexp_replace(text, r"[\n\r]\s*[\n\r]+", "\n")  # collapse multiple newlines

    # Trim overall
    return F.regexp_replace(text, r"^[ \n\r\t]+|[ \n\r\t]+$", "")


def run_parser(spark, file_path: str) -> DataFrame:
    print(">>> [1/4] ETL: Parsing & Cleaning ...")

    if not spark or not file_path:
        print(">>> ERROR: Invalid Spark session or file path.")
        return None

    try:
        # Load XML
        df_raw = spark.read.format("xml").option("rowTag", "page").load(file_path)

        # --- Column Selection ---
        title_col = _pick_col(df_raw, ["title"])
        ns_col = _pick_col(df_raw, ["ns", "ns._VALUE"])
        id_col = _pick_col(df_raw, ["id", "id._VALUE"])
        redirect_col = _pick_col(df_raw, ["redirect._title", "redirect._VALUE", "redirect"]) 
        revision_id_col = _pick_col(df_raw, ["revision.id", "revision.id._VALUE"])
        timestamp_col = _pick_col(df_raw, ["revision.timestamp", "revision.timestamp._VALUE"])
        text_col = pick_text_col(df_raw)

        select_exprs = []
        if title_col is not None: select_exprs.append(F.trim(title_col.cast("string")).alias("title"))
        if ns_col is not None: select_exprs.append(ns_col.cast("int").alias("ns"))
        if id_col is not None: select_exprs.append(id_col.cast("long").alias("id"))
        if redirect_col is not None: select_exprs.append(redirect_col.cast("string").alias("redirect"))
        if revision_id_col is not None: select_exprs.append(revision_id_col.cast("long").alias("revision_id"))
        if timestamp_col is not None: select_exprs.append(timestamp_col.cast("string").alias("timestamp"))
        if text_col is not None: select_exprs.append(text_col.cast("string").alias("text_raw"))

        if not select_exprs:
            print(">>> ERROR: No valid columns found.")
            return df_raw

        df = df_raw.select(*select_exprs)

        # --- Extraction & Cleaning ---
        if "text_raw" in df.columns:
            empty_arr = F.expr("cast(array() as array<string>)")
            text_raw = F.coalesce(F.col("text_raw"), F.lit(""))

            # Helper: Trims edges but preserves internal spaces (e.g., "New York")
            def clean_array(col):
                return F.array_distinct(F.filter(
                    F.transform(col, lambda x: F.trim(x)),
                    lambda x: x.isNotNull() & (x != "")
                ))

            # 1. Regex Extraction
            categories = F.regexp_extract_all(text_raw, F.lit(r"\[\[(?i:Category):([^\]\|]+)"), 1)
            templates = F.regexp_extract_all(text_raw, F.lit(r"\{\{\s*([^\|\}\n<]+)"), 1)
            
            # External Links: Capture full link text until ']', newline, or tag start
            external_links = F.regexp_extract_all(text_raw, F.lit(r"(https?://[^\]\|\n<>]+)"), 1)
            
            # Internal Links: Capture target and strip leading colons
            links_raw = F.regexp_extract_all(text_raw, F.lit(r"\[\[([^\]\|#]+)"), 1)
            links_norm = F.transform(links_raw, lambda x: F.regexp_replace(x, r"^:+", ""))
            
            # 2. Apply Transformations
            df = (
                df
                .withColumn("categories", F.coalesce(clean_array(categories), empty_arr))
                .withColumn("templates", F.coalesce(clean_array(templates), empty_arr))
                .withColumn("external_links", F.coalesce(clean_array(external_links), empty_arr))
                .withColumn("links", F.coalesce(clean_array(links_norm), empty_arr))
                # Filter special namespaces from links list
                .withColumn("links", F.filter("links", lambda x: 
                    (~F.lower(x).startswith("category:")) & 
                    (~F.lower(x).startswith("file:")) & 
                    (~F.lower(x).startswith("image:"))
                ))
                .withColumn("text", _clean_wikitext(F.col("text_raw")))
            )

            print("\n>>> Parser Sample Check")
            sample_rows = (
                df
                .where(F.col("text_raw").isNotNull())
                .where(F.col("ns") == 0)
                .limit(1)
                .collect()
            )
            
            if sample_rows:
                row = sample_rows[0]
                print(f"--- ROW DATA ---")
                print(f"1. ID:             {row['id']}")
                print(f"2. Title:          {row['title']}")
                print(f"3. Namespace:      {row['ns']}")
                print(f"4. Revision ID:    {row['revision_id']}")
                print(f"5. Timestamp:      {row['timestamp']}")
                print(f"6. Redirect:       {row['redirect']}")
                print(f"7. Templates:      {row['templates']}")
                print(f"8. Categories:     {row['categories']}")
                print(f"9. External Links: {row['external_links']}")
                print(f"10.Internal Links: {row['links']}")
                print(f"11.Text length:    {len(row['text']) if row['text'] else 0}")
                print(f"12.Clean Text:\n{row['text']}")
                print(f"----------------")
                print(f"13.Raw Text:\n{row['text_raw'][0:500]}...")
            else:
                print(">>> Warning: No valid rows found for validation.")

        return df

    except Exception as e:
        print(f">>> ERROR: Parser execution failed: {e}")
        return None