import pip

kREQ_PKG = ['unidecode', 'wikipedia', 'whoosh', 'nltk', 'scikit-learn', 'regex']

def install(package):
    pip.main(['install', package])

if __name__ == '__main__':
    for ii in kREQ_PKG:
        install(ii)

    import nltk
    nltk.download("stopwords")
    nltk.download('punkt')
