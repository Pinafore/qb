import pip

kREQ_PKG = ['unidecode', 'wikipedia', 'whoosh', 'nltk', 'scikit-learn', 'regex', 'fuzzywuzzy', 'python-Levenshtein', 'kenlm', 'pattern']


def install(package):
    pip.main(['install', package])

if __name__ == '__main__':
    for ii in kREQ_PKG:
        install(ii)
