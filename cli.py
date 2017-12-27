import click

from qanta import qlogging
from qanta.util.environment import ENVIRONMENT
from qanta.wikipedia.cached_wikipedia import web_initialize_file_cache


log = qlogging.get(__name__)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
@click.argument('wiki_cache')
def init_wiki_cache(wiki_cache):
    web_initialize_file_cache(wiki_cache)


if __name__ == '__main__':
    main()
