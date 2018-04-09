import os
from setuptools import setup, find_packages, Command


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as output:
        return output.read()

requirements = [
    'scipy',
    'numpy',
    'nltk',
    'scikit-learn',
    'fuzzywuzzy',
    'python-Levenshtein',
    'requests',
    'click',
    'pyfunctional',
    'luigi',
    'jinja2',
    'progressbar2',
    'boto3',
    'matplotlib',
    'spacy',
    'pycountry',
    'flask',
    'dash',
    'dash-renderer',
    'dash-html-components',
    'dash-core-components',
    'plotly',
    'unidecode',
    'pyyaml'
]


class DownloadCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        path = 'data/external/nltk_download_SUCCESS'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('Downloaded nltk: stopwords, pinkt, wordnet')

setup(
    name='qb',
    version='2.0.0',
    description='Quiz Bowl AI system named QANTA',
    license='MIT',
    long_description=read('README.md'),
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    cmdclass={'download': DownloadCommand}
)
