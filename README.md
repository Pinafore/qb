# QANTA

## Setups
0. You'll need Python 3.5 or later, R, Gorobi, Apache Spark, Rust 1.7, and Vowpal Wabbit installed.

1. Either copy non_naqt.db to data/questions.db, simlink it, or copy your own questions.db file.

2. Run the script "python util/install_python_packages.py", which will install several python packages you'll need.  (You may need admin access.)

3. Run the script "python util/install_nltk_data.py", which will download some nltk data.  You should *not* use admin access for this script.

4. Download the Illinois Wikifier code (VERSION 2).  Place the data directory in data/wikifier/data and put the wikifier-3.0-jar-with-dependencies.jar in the lib directory http://cogcomp.cs.illinois.edu/page/software_view/Wikifier and put the config directory in data/wikifier/config

## Environment Variables
To make running the QB software easier, certain environment variables are used to set data directories. Below are a list of the variables and what they currently work for

```
# Variables used by run_spark.py
export QB_QUESTION_DB=data/non_naqt.db
export QB_GUESS_DB=data/guesses.db
export QB_SPARK_MASTER=spark-master-url
```

## Spark Configuration
Be sure to edit `SPARK`

## Steps
1.  Generate the Makefile

    ``python generate_makefile.py``

2.  Generate the guess database (this takes a while, depends on DAN---60
hours---and guesses---40 hours)

    ``make data/guesses.db``

3.  Generate the LM pickle (18 hours)

    ``make data/lm.pkl``

4. generate features, train all models, and get predictions.

    ``make all_sentence_buzz``

5. answer the questions found in data/expo.csv and run the demo

    ``make demo4``

Feature timings:

      *  classifier: 216 features lines per sec
      *  lm: 139.028408 feature lines per sec
      *  deep: 84.391876 feature lines per sec
      *  text: 158.384899 feature lines per sec
      *  wikilinks: 62.842486 feature lines per sec
      *  answer_present: 155.469810 feature lines per sec

## Output File
In the directory `data/results/` there are a number of files and folders which are the output of a quiz bowl run. Below is a description of what each file contains:

* `sentence.X.full.final`: the answer for a question given the full question text
* `sentence.X.full.buzz`: For each (question, sentence, token), the guess with whether to buzz and the weight
* `sentence.X.full.perf`: For each (question, sentence, token), performance statistics
* `sentence.X.full.pred`: Prediction weights from vowpal wabbit
