from pyspark.sql import SparkSession


def get_spark(app_name="WikiDump-Spark",
              num_threads=8):
    """Create and return a SparkSession.
    Args:
        app_name (str): Name of the Spark application.
        num_threads (int): Number of local threads to use.
    """
    try:
        # builder = SparkSession.builder.appName(app_name)
        # master = os.getenv("SPARK_MASTER")
        # if master:
        #     builder = builder.master(master)

        # spark = builder.getOrCreate()
        # if spark is None:
        #     raise RuntimeError("Failed to create SparkSession")
        # return spark
        builder = SparkSession.builder.appName(app_name)
        builder = builder.master(f"local[{num_threads}]")
        spark = builder.getOrCreate()
        if spark is None:
            raise RuntimeError("Failed to create SparkSession")
        return spark
    except Exception as e:
        print(f">>> ERROR: Failed to initialize Spark: {e}")
        raise