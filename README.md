# QANTA

Setups
-----------
0.  You'll need Python, R, and Vowpal Wabbit installed and accessible on the
    path.

1.  Make sure you have the question database, store it as data/questions.db

2.  Run the script "python util/install_python_packages.py", which will install
several python packages you'll need.  (You may need admin access.)  This will
also download some nltk data.

3. Download the deep model / classifier here: https://www.dropbox.com/s/ly51cxnak0fqth5/deep.tar.gz?dl=0 and extract it to data/deep/

Steps
-----------

1.  Generate the Makefile

``python generate_makefile.py``

2.  Generate the guess database

``make data/guesses.db``

3.  Generate the LM pickle

``make data/lm.pkl``

4. generate features, train all models, and get predictions

``make all_sentence_buzz``



