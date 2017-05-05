# Root directory of Quiz Bowl repository
export QB_ROOT=/ssd-c/qanta/qb

# URL of the Apache Spark cluster master
export QB_SPARK_MASTER=localhost

# Number of Spark cores to use for streaming
export QB_STREAMING_CORES=12

# Parameters for interacting with the Shared Task API server
export QB_API_DOMAIN=""
export QB_API_USER_ID=1
export QB_API_KEY=""

export PYTHONPATH=$PYTHONPATH:/ssd-c/qanta/qb
