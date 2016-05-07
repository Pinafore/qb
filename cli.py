import click

from qanta.spark_execution import start_streaming


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    pass


@main.command(name='stream')
def stream():
    start_streaming()


if __name__ == '__main__':
    main()
