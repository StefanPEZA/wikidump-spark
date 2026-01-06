from jobs.config import get_spark
from jobs.parser import run_parser
from jobs.stats  import run_stats
from jobs.nlp    import run_nlp
from jobs.graph  import run_graph

DATA_FILE = "data/enwiki-latest-pages-articles-multistream11.xml-p6899367p7054859"

def main():
    spark = get_spark()

    try:
        print(">>> Starting Pipeline...")
        
        # ETL: Parsing & Cleaning...
        df = run_parser(spark, DATA_FILE)
        if df is None:
            return
        
        print(">>> Schema:")
        df.printSchema()
        
        df.cache()
        
        # EDA: Statistics...
        run_stats(df)
        
        # NLP: Topic Modeling...
        run_nlp(df)
        
        # Graph: Extracting Edges...
        run_graph(df)

    finally:
        spark.stop()
        print(">>> Spark Session Stopped.")

if __name__ == "__main__":
    main()