import luigi
from qanta.util.io import shell
from qanta.guesser.elasticsearch_wikidata import WIKIDATA_PICKLE


class DownloadWikidata(luigi.Task):
    def run(self):
        shell('aws s3 cp s3://entilzha-us-west-2/wikidata/wikidata.pickle {}'.format(WIKIDATA_PICKLE))

    def output(self):
        return luigi.LocalTarget(WIKIDATA_PICKLE)
