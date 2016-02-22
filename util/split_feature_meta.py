import sys
import click
import re


@click.command()
@click.argument('feat_destination', type=click.File('w'))
@click.argument('meta_destination', type=click.File('w'))
def main(feat_destination, meta_destination):
    delimiter = re.compile('\|\|\|')
    for line in sys.stdin:
        feat, meta = re.split(delimiter, line)
        feat_destination.write(feat + '\n')
        meta_destination.write(meta + '\n')


if __name__ == '__main__':
    main()
