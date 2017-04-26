import click

from qanta import logging
from qanta.util.environment import ENVIRONMENT
from qanta.util.vw import format_audit
from qanta.wikipedia.cached_wikipedia import web_initialize_file_cache


log = logging.get(__name__)

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


@main.command()
@click.option('--n_features', type=int, default=20)
def format_vw_audit(n_features):
    format_audit(n_features)

if __name__ == '__main__':
    main()
