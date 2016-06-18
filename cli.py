import click

from qanta.extractors import mentions
from qanta.streaming import start_qanta_streaming, start_spark_streaming


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command()
def spark_stream():
    start_spark_streaming()


@main.command()
def qanta_stream():
    start_qanta_streaming()


@main.command()
def train_clm():
    pass


@main.command()
@click.argument('wikipedia_input')
@click.argument('output')
def build_mentions_lm_data(wikipedia_input, output):
    mentions.build_lm_data(wikipedia_input, output)

if __name__ == '__main__':
    main()
