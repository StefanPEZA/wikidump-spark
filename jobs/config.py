from pyspark.sql import SparkSession

def get_spark():
    try:
        spark = SparkSession.builder.appName("WikiDump-Spark").getOrCreate()
        if spark is None:
            raise RuntimeError("Failed to create SparkSession")
        return spark
    except Exception as e:
        print(f">>> ERROR: Failed to initialize Spark: {e}")
        raise