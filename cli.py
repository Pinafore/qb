# Script to complete targets as the first argument
#
# If run directly, iterates through all the locations on the ENVIRONMENT variable and
# makes sure they're up-to-date.

import click

from qanta import logging
from qanta.extractors import mentions
from qanta.util.environment import ENVIRONMENT
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.wikipedia import wikification
from qanta.streaming import start_qanta_streaming, start_spark_streaming


log = logging.get(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
def spark_stream():
    start_spark_streaming()


@main.command()
def qanta_stream():
    start_qanta_streaming()


@main.command()
@click.argument('wikipedia_input')
@click.argument('output')
def build_mentions_lm_data(wikipedia_input, output):
    mentions.build_lm_data(wikipedia_input, output)


@main.command()
@click.argument('wiki_cache')
def init_wiki_cache(wiki_cache):
    CachedWikipedia.initialize_cache(wiki_cache)


@main.command()
@click.argument('output')
def wikify(output):
    wikification.wikify(output)

if __name__ == '__main__':
    main()
