# Requirements
NLTK version 3.0.2 or higher
NumPy version 1.9.2 or higher
Pretrained GloVe vectors: http://www-nlp.stanford.edu/data/glove.840B.300d.txt.gz (put in data folder)
Also needs NER pickle and questions database 

# Training Deep Guesser
cd util
python format_dan.py
python load_embeddings.py
cd ..
python dan.py (this step may take 15 hours or more depending on dataset size)