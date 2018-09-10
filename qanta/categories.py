"""
Process Wikipedia category links
"""
import json
import csv
import click


@click.group()
def categorylinks_cli():
    pass


@categorylinks_cli.command()
@click.argument('categories_csv')
@click.argument('out_jsonl')
def clean(categories_csv, out_jsonl):
    out_lines = []
    with open(categories_csv) as f:
        for line in csv.reader(f):
            if len(line) == 2:
                out_lines.append(json.dumps({
                    'id': int(line[0]),
                    'cat': line[1]
                    }))
    with open(out_jsonl, 'w') as f:
        for l in out_lines:
            f.write(l)
            f.write('\n')
