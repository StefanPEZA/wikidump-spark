import os
import platform
import sys

from pyspark import SparkConf
from pyspark.sql import SparkSession


def get_spark(app_name="WikiDump-Spark",
              num_threads=8,
              driver_memory: str | None = None,
              executor_memory: str | None = None,
              driver_max_result_size: str | None = None):
    """Create and return a SparkSession.
    Args:
        app_name (str): Name of the Spark application.
        num_threads (int): Number of local threads to use.
        driver_memory (str|None): e.g. "4g", "8g".
        executor_memory (str|None): e.g. "4g", "8g".
        driver_max_result_size (str|None): e.g. "1g".
    """
    try:
        py = sys.executable
        os.environ["PYSPARK_PYTHON"] = py
        os.environ["PYSPARK_DRIVER_PYTHON"] = py

        conf_defaults = SparkConf(loadDefaults=True)
        submit_master = conf_defaults.get("spark.master", None)
        env_master = os.getenv("MASTER")
        local_fallback_master = f"local[{num_threads}]"  # e.g. python main.py ...

        builder = SparkSession.builder.appName(app_name)

        if env_master:
            builder = builder.master(env_master)
        elif not submit_master:
            builder = builder.master(local_fallback_master)

        # Memory configuration
        driver_memory = driver_memory or os.getenv("DRIVER_MEMORY")
        executor_memory = executor_memory or os.getenv("EXECUTOR_MEMORY")
        driver_max_result_size = driver_max_result_size or os.getenv("DRIVER_MAX_RESULT_SIZE")

        if driver_memory is None and conf_defaults.get("spark.driver.memory", None) is None:
            driver_memory = "8g"
        if executor_memory is None:
            if conf_defaults.get("spark.executor.memory", None) is None:
                executor_memory = driver_memory
        if driver_max_result_size is None and conf_defaults.get("spark.driver.maxResultSize", None) is None:
            driver_max_result_size = "4g"

        if driver_memory is not None:
            builder = builder.config("spark.driver.memory", driver_memory)
        if executor_memory is not None:
            builder = builder.config("spark.executor.memory", executor_memory)
        if driver_max_result_size is not None:
            builder = builder.config("spark.driver.maxResultSize", driver_max_result_size)

        # Ensure Spark uses the same Python interpreter as the current process.
        builder = (
            builder
            .config("spark.pyspark.python", py)
            .config("spark.pyspark.driver.python", py)
            .config("spark.executorEnv.PYSPARK_PYTHON", py)
        )

        env_driver_host = os.getenv("DRIVER_HOST")
        env_driver_bind = os.getenv("DRIVER_BIND_ADDRESS")

        effective_master = env_master or submit_master or local_fallback_master
        is_local_master = str(effective_master).startswith("local")
        if is_local_master:
            env_driver_host = env_driver_host or "127.0.0.1"
            env_driver_bind = env_driver_bind or "127.0.0.1"

        if env_driver_bind:
            builder = builder.config("spark.driver.bindAddress", env_driver_bind)
        if env_driver_host:
            builder = builder.config("spark.driver.host", env_driver_host)

        # GraphFrames package
        graphframes_pkg = os.getenv("GRAPHFRAMES_PACKAGE")
        builder = builder.config(
            "spark.jars.packages",
            graphframes_pkg or "io.graphframes:graphframes-spark4_2.13:0.10.0",
        )
        
        spark = builder.getOrCreate()
        if spark is None:
            raise RuntimeError("Failed to create SparkSession")
        return spark
    except Exception as e:
        print(f">>> ERROR: Failed to initialize Spark: {e}")
        raise