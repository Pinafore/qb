# QANTA

## Downloading Data

Whether you would like to use our system or use only our dataset, the easiest way to do so is
use our `dataset.py` script. It is a standalone script whose only dependencies are python 3.6 and the package `click`
which can be installed via `pip install click`.

The following commands can be used to download our dataset, or datasets we use in either the system or paper plots.
Data will be downloaded to `data/external/datasets` by default, but can be changed with the `--local-qanta-prefix`
option

* `./dataset.py download`: Download only the qanta dataset
* `./dataset.py download wikidata`: Download our preprocessed wikidata.org `instance of` attributes
* `./dataset.py download plotting`: Download the squad, simple questions, jeopardy, and triviaqa datasets we
compare against in our paper plots and tables

### File Description:

* `qanta.unmapped.2018.04.18.json`: All questions in our dataset, without mapped Wikipedia answers. Sourced from
protobowl and quizdb. Light preprocessing has been applied to remove quiz bowl specific syntax such as instructions
to moderators
* `qanta.processed.2018.04.18.json`: Prior dataset with added fields extracting the first sentence, and sentence tokenizations
of the question paragraph for convenience.
* `qanta.mapped.2018.04.18.json`: The processed dataset with Wikipedia pages matched to the answer where possible. This
includes all questions, even those without matched pages.
* `qanta.2018.04.18.sqlite3`: Equivalent to `qanta.mapped.2018.04.18.json` but in sqlite3 format
* `qanta.train.2018.04.18.json`: Training data which is the mapped dataset filtered down to only questions with non-null
page matches
* `qanta.dev.2018.04.18.json`: Dev data which is the mapped dataset filtered down to only questions with non-null
page matches
* `qanta.test.2018.04.18.json`: Test data which is the mapped dataset filtered down to only questions with non-null
page matches

## Dependencies

The recommended way to run our system is to use the Anaconda python
distribution. The `environment.yaml` can be used to create a `conda`
environment with all the necessary software versions installed.

The qanta system has the following dependencies. Depending on your objective
however not all are necessary. The python packages are generally required so
that imports resolve, Apache Spark is required for many data preprocessing
steps, Vowpal Wabbit is not needed for anything except training a linear model,
Spacy is required for preprocessing, Elastic Search is required for the IR
based models, and `lz4` and the AWS cli are necessary for downloading data not
part of the `dataset.py` script.

* Anaconda Distributed Python 3.6
* PyTorch 0.3.X
* Apache Spark 2.2.0 with Scala/JVM
* Vowpal Wabbit 8.4
* Spacy 2.0
* Elastic Search 5.6.X
* lz4
* AWS CLI
* All python packages listed in `environment.yaml`

### Installing Apache Spark

`packer/bin/install-spark.sh`

You can test is spark is installed property by running something like the following:
```
> python
Python 3.6.1 |Anaconda 4.4.0 (64-bit)| (default, May 11 2017, 13:09:58) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from qanta.spark import create_spark_context
>>> sc = create_spark_context()
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
17/07/25 10:04:01 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
17/07/25 10:04:01 WARN Utils: Your hostname, hongwu resolves to a loopback address: 127.0.0.2; using 192.168.2.2 instead (on interface eth0)
17/07/25 10:04:01 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
```

### Installing Elastic Search 5.6

`packer/bin/install-elasticsearch.sh`

Install version 5.6.X, do not use 6.X. Also be sure that the directory `bin/`
within the extracted files is in your `$PATH` as it contains the necessary
binary `elasticsearch`.


### Installing Python packages

Either use `environment.yaml` or:

```
pip install -r packer/requirements.txt
```

#### NLTK Models
```bash
# Download nltk data
$ python3 setup.py download
```

### Qanta on Path

In addition to these steps you need to either run `python setup.py develop` or
include the qanta directory in your `PYTHONPATH` environment variable. We
intend to fix path issues in the future by fixing absolute/relative paths.


## Configuration

QANTA configuration is done through a combination of environment variables and
the `qanta-defaults.yaml`/`qanta.yaml` files. QANTA will read a `qanta.yaml`
first if it exists, otherwise it will fall back to reading
`qanta-defaults.yaml`. This is meant to allow for custom configuration of
`qanta.yaml` after copying it via `cp qanta-defaults.yaml qanta.yaml`.

The configuration of most interest is how to enable or disable specific guesser
implementations. In the `guesser` config the keys such as
`qanta.guesser.dan.DanGuesser` correspond to the fully qualified paths of each
guesser. Each of these keys contain an array of configurations (this is
signified in yaml by the `-`). Our code will inspect all of these
configurations looking for those that have `enabled: true`, and only run those
guessers. By default we have `enabled: false` for all models. If you simply
want to perform a sanity check we recommend enabling
`qanta.guesser.tfidf.TfidfGuesser`. If you are looking for our best model and
configuration you should use enable `qanta.guesser.rnn.RnnGuesser`.

## Running QANTA

Running qanta is managed primarily by two methods: `./cli.py` and
[Luigi](https://github.com/spotify/luigi). The former is used to run specific
commands such as starting/stopping elastic search, but in general `luigi` is
the primary method for running our system.

### Luigi Pipelines
Luigi is a pure python make-like framework for running data pipelines. Below we
give sample commands for running different parts of our pipeline. In general,
you should either append `--local-scheduler` to all commands or learn about
using the [Luigi Central
Scheduler](https://luigi.readthedocs.io/en/stable/central_scheduler.html).

For these common tasks you can use command `luigi --local-scheduler` followed by:

* `--module qanta.pipeline.preprocess DownloadData`: This downloads any
  necessary data and preprocesses it. This will download a copy of our
preprocessed Wikipedia stored in AWS S3 and turn it into the format used by our
code. This step requires the AWS CLI, `lz4`, Apache Spark, and may require a
decent amount of RAM.
* `--module qanta.pipeline.guesser AllGuesserReports`: Train all enabled
  guessers, generate guesses for them, and produce a report of their
performance into `output/guesser`.



### Qanta CLI

You can start/stop elastic search with
* `./cli.py elasticsearch start`
* `./cli.py elasticsearch stop`


## AWS S3 Checkpoint/Restore

To provide and easy way to version, checkpoint, and restore runs of qanta we provide a script to
manage that at `aws_checkpoint.py`. We assume that you set an environment variable
`QB_AWS_S3_BUCKET` to where you want to checkpoint to and restore from. We assume that we have full
access to all the contents of the bucket so we suggest creating a dedicated bucket.

## Information on our data sources
### Wikipedia Dumps

As part of our ingestion pipeline we access raw wikipedia dumps. The current code is based on the english wikipedia
dumps created on 2017/04/01 available at https://dumps.wikimedia.org/enwiki/20170401/

Of these we use the following (you may need to use more recent dumps)

* [Wikipedia page text](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-pages-articles-multistream.xml.bz2): This is used to get the text, title, and id of wikipedia pages
* [Wikipedia titles](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-all-titles.gz): This is used for more convenient access to wikipedia page titles
* [Wikipedia redirects](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-redirect.sql.gz): DB dump for wikipedia redirects, used for resolving different ways of referencing the same wikipedia entity
* [Wikipedia page to ids](https://dumps.wikimedia.org/enwiki/20170401/enwiki-20170401-page.sql.gz): Contains a mapping of wikipedia page and ids, necessary for making the redirect table useful

To process wikipedia we use [https://github.com/attardi/wikiextractor](https://github.com/attardi/wikiextractor)
with the following command:

```bash
$ WikiExtractor.py --processes 15 -o parsed-wiki --json enwiki-20170401-pages-articles-multistream.xml.bz2
```

Do not use the flag to filter disambiguation pages. It uses a simple string regex to check the title and articles contents. This introduces both false positives and false negatives. We handle the problem of filtering these out by using the wikipedia categories dump

Afterwards we use the following command to tar it, compress it with lz4, and upload the archive to S3

```bash
tar cvf - parsed-wiki | lz4 - parsed-wiki.tar.lz4
```

#### Wikipedia Redirect Mapping Creation

The output of this process is stored in `s3://pinafore-us-west-2/public/wiki_redirects.csv`

All the wikipedia database dumps are provided in MySQL sql files. This guide has a good explanation of how to install MySQL which is necessary to use SQL dumps. For this task we will need these tables:

* Redirect table: https://www.mediawiki.org/wiki/Manual:Redirect_table
* Page table: https://www.mediawiki.org/wiki/Manual:Page_table
* The namespace page is also helpful: https://www.mediawiki.org/wiki/Manual:Namespace

To install, prepare MySQL, and read in the Wikipedia SQL dumps execute the following:

1. Install MySQL `sudo apt-get install mysql-server` and `sudo mysql_secure_installation`
2. Login with something like `mysql --user=root --password=something`
3. Create a database and use it with `create database wikipedia;` and `use wikipedia;`
4. `source enwiki-20170401-redirect.sql;` (in MySQL session)
5. `source enwiki-20170401-page.sql;` (in MySQL session)
6. This will take quite a long time, so wait it out...
7. Finally run the query to fetch the redirect mapping and write it to a CSV by executing `bin/redirect.sql` with `source bin/redirect.sql`. The file will be located in `/var/lib/mysql/redirect.csv` which requires `sudo` access to copy
8. The result of that query is CSV file containing a source page id, source page title, and target page title. This can be
interpretted as the source page redirecting to the target page. We filter namespace=0 to keep only redirects/pages that are main pages and trash things like list/category pages

#### Wikipedia Category Links Creation

The purpose of this step is to use wikipedia category links to filter out disambiguation pages. Every wikipedia page
has a list of categories it belongs to. We filter out any pages which have a category which includes the string `disambiguation`
in its name. The output of this process is a json file containing a list of page_ids that correspond to known disambiguation pages.
These are then used downstream to filter down to only non-disambiguation wikipedia pages.

The output of this process is stored in `s3://pinafore-us-west-2/public/disambiguation_pages.json` with the csv also
saved at `s3://pinafore-us-west-2/public/categorylinks.csv`

The process for this is similar to redirects, except that you should instead source a file named similar to `enwiki-20170401-categorylinks.sql`, run
the script `bin/categories.sql`, and copy `categorylinks.csv`. Afterwards run `./cli.py categories disambiguate categorylinks.csv data/external/wikipedia/disambiguation_pages.json`.
This file is automatically downloaded by the pipeline code like the redirects file so unless you would like to change this or inspect the results, you shouldn't need to worry about this.

##### SQL References

These references may be useful and are the source for these instructions:

* https://www.digitalocean.com/community/tutorials/how-to-install-mysql-on-ubuntu-16-04
* https://dev.mysql.com/doc/refman/5.7/en/mysql-batch-commands.html
* http://stackoverflow.com/questions/356578/how-to-output-mysql-query-results-in-csv-format

## Debugging FAQ and Solutions

> pyspark uses the wrong version of python

Set PYSPARK_PYTHON to be python3

> ImportError: No module named 'pyspark'

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH

> ValueError: unknown locale: UTF-8

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

> TypeError: namedtuple() missing 3 required keyword-only arguments: 'verbose', 'rename', and 'module'

Python 3.6 needs Spark 2.1.1
