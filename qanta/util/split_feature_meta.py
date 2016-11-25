import sys
import click
import re


@click.command()
@click.argument('feat_destination', type=click.Path())
@click.argument('meta_destination', type=click.Path())
def main(feat_destination, meta_destination):
    delimiter = re.compile('@')
    feat_file = open(feat_destination, 'a')
    meta_file = open(meta_destination, 'a')
    for line in sys.stdin:
        feat, meta = re.split(delimiter, line)
        print(feat.strip(), file=feat_file)
        print(meta.strip(), file=meta_file)

    feat_file.close()
    meta_file.close()


if __name__ == '__main__':
    main()
