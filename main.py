from jobs.config import get_spark
from jobs.parser import run_parser
from jobs.stats  import run_stats
from jobs.nlp    import run_nlp
from jobs.graph  import run_graph

DATA_FILE = "data/enwiki-latest-pages-articles-multistream.xml.bz2"

def main():
    spark = get_spark()

    try:
        print(">>> Starting Pipeline...")
        
        # ETL: Parsing & Cleaning...
        df_clean = run_parser(spark, DATA_FILE)
        if df_clean is None:
            return
        
        df_clean.cache()
        
        # EDA: Statistics...
        run_stats(df_clean)
        
        # NLP: Topic Modeling...
        run_nlp(df_clean)
        
        # Graph: Extracting Edges...
        run_graph(df_clean)

    finally:
        spark.stop()
        print(">>> Spark Session Stopped.")

if __name__ == "__main__":
    main()