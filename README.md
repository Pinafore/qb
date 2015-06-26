# QANTA

Setups
-----------
0.  You'll need Python, R, Gorobi, and Vowpal Wabbit installed and accessible on the
    path.

1.  Make sure you have the question database, store it as data/questions.db

2.  Run the script "python util/install_python_packages.py", which will install
several python packages you'll need.  (You may need admin access.)  This will
also download some nltk data.

3. Download the Illinois Wikifier code.  Place the data directory in
   data/wikifier/data and put the wikifier-3.0-jar-with-dependencies.jar in the lib
   directory.

http://cogcomp.cs.illinois.edu/page/software_view/Wikifier

4.  Copy the non_naqt.db to data/questions.db (or make a symbolic link)

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
