# Exercise 7

1. Background

The purpose of this exercise is to teach you to use Spark for conducting distributed machine learning computation.

2. The task

Your task is the Bosch challenge (https://www.kaggle.com/c/bosch-production-line-performance) from Kaggle. This is a binary classification task, with a rather imbalanced response variable.

You can fetch the data either from the Kaggle site, or from an S3 bucket (s3://mle7-data, available from within sandbox AWS account). The response variable is included in the file train_numerical.csv. Feel free to start with only numerical variables, and include categorical and date variables later.

Notice that the test data does not include the response variable. If you wish to evaluate your model, you can do it by splitting your training data into test and train subsets.

Do not spent too much time on model improvement and evaluation -- it is more important that you learn to do machine learning with Spark.

3. Howto

You can get to know Spark by downloading it via the downloads page: http://spark.apache.org/downloads.html. Easiest way to try it out is perhaps via Python bindings: 

```shell
> pip install pyspark
...
> pyspark
Python 2.7.14 (default, Mar 22 2018, 14:43:05)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.39.2)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
2018-09-06 21:41:23 WARN  NativeCodeLoader:62 - Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.3.1
      /_/

Using Python version 2.7.14 (default, Mar 22 2018 14:43:05)
SparkSession available as 'spark'.
>>> sc.parallelize(range(100)).sum()
4950  
```

After starting Spark (say via `pyspark`), the management UI of a running Spark instance is available at http://localhost:4040/. Browsing it may give insight on the job execution of your code.

There will also be a EMR cluster set up for running Spark. The cluster will be available via SSH from a later announced address. `pyspark` will also be available.
