import os
from setuptools import setup, find_packages, Command


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as output:
        return output.read()

requirements = [
    'scipy',
    'numpy',
    'unidecode',
    'wikipedia',
    'whoosh',
    'nltk',
    'scikit-learn',
    'regex',
    'fuzzywuzzy',
    'py4j',
    'python-Levenshtein',
    'requests',
    'click',
    'pyfunctional',
    'luigi',
    'jinja2'
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

setup(
    name='qb',
    version='0.0.1',
    description='Quiz Bowl AI system named QANTA',
    license='MIT',
    long_description=read('README.md'),
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    cmdclass={'download': DownloadCommand}
)
