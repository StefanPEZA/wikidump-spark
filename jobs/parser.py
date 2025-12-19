
def run_parser(spark, file_path):
    print(">>> [1/4] ETL: Parsing & Cleaning...")

    if spark is None:
        print(">>> ERROR: Spark session is None")
        return None
    if not file_path:
        print(">>> ERROR: File path is empty")
        return None

    try:
        df = spark.read.format("xml").option("rowTag", "page").load(file_path)

        try:
            # Extracting and printing article titles
            for row in df.select("title").toLocalIterator():
                title = None
                if hasattr(row, '__contains__') and 'title' in row:
                    title = row['title']
                    print("Title [" + title + "]")
        except Exception as e:
            print(f">>> ERROR: Printing titles failed: {e}")

        return df
    except Exception as e:
        print(f">>> ERROR: Parser failed: {e}")
        return None
