# QANTA

## Setups
0.  You'll need Python 3.5 or later, R, Gorobi, Apache Spark, and Vowpal Wabbit installed.

1. Either copy non_naqt.db to data/questions.db, simlink it, or copy your own questions.db file.

2.  Run the script "python util/install_python_packages.py", which will install
several python packages you'll need.  (You may need admin access.)  

3.  Run the script "python util/install_nltk_data.py", which will download
some nltk data.  You should *not* use admin access for this script.


4. Download the Illinois Wikifier code (VERSION 2).  Place the data directory in
   data/wikifier/data and put the wikifier-3.0-jar-with-dependencies.jar in the lib
   directory.

http://cogcomp.cs.illinois.edu/page/software_view/Wikifier

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

## Steps for quick test
If you are interested in getting the qb system running end to end without training the full system,
you can follow these steps.

1. Generate the Makefile like above
2. Run `make data/deep/glove.840B.300d.txt.gz` to download some data

