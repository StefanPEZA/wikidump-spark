import argparse
from jobs.config import get_spark
from jobs.parser import run_parser
from jobs.stats  import run_stats
from jobs.nlp    import run_nlp
from jobs.graph  import run_graph

## python main.py --input "./data/enwiki-latest-pages-articles-multistream11.xml-p6899367p7054859" --out "./out/local-run" --k 10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path (HDFS or local path)")
    ap.add_argument("--out", required=True, help="Folder output (HDFS or local path)")
    ap.add_argument("--k", type=int, default=10, help="Number of KMeans/LDA clusters")
    args = ap.parse_args()

    spark = get_spark()

    try:
        print(">>> Starting Pipeline...")

        # ETL: Parsing & Cleaning...
        df = run_parser(spark, args.input)
        if df is None:
            return

        print(">>> Schema:")
        df.printSchema()

        df.cache()

        # EDA: Statistics...
        run_stats(df, args.out)

        # NLP: Topic Modeling...
        run_nlp(df, args.out, k=args.k)

        # Graph: Extracting Edges...
        run_graph(df, args.out)

    finally:
        spark.stop()
        print(">>> Spark Session Stopped.")

if __name__ == "__main__":
    main()