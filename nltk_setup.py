import os
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
path = 'data/external/nltk_download_SUCCESS'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as f:
    f.write('Downloaded nltk: stopwords, punkt, wordnet')
