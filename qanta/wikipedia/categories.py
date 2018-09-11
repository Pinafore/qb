"""
Process Wikipedia category links
"""
import json
import re
import csv
import click
import tqdm


@click.group()
def categorylinks_cli():
    pass


@categorylinks_cli.command()
@click.argument('categories_csv')
@click.argument('out_jsonl')
def clean(categories_csv, out_jsonl):
    with open(categories_csv) as in_f, open(out_jsonl, 'w') as out_f:
        for line in csv.reader(in_f):
            if len(line) == 2:
                if re.match(r'[a-zA-Z0-9\-\_\s]+', line[1]):
                    out_f.write(json.dumps({
                        'id': int(line[0]),
                        'cat': line[1]
                        }))
                    out_f.write('\n')

@categorylinks_cli.command()
@click.argument('category_csv')
@click.argument('out_json')
def disambiguate(category_csv, out_json):
    disambiguation_pages = set()
    blacklist = {
        'Articles_with_links_needing_disambiguation_from_April_2018',
        'All_articles_with_links_needing_disambiguation'
    }
    with open(category_csv) as f:
        reader = csv.reader(f)
        for r in tqdm.tqdm(reader, mininterval=1):
            page_id, category = r[0], r[1]
            l_category = category.lower()
            if ((category not in blacklist) and
                    ('disambiguation' in l_category) and
                    ('articles_with_links_needing_disambiguation' not in l_category)):
                disambiguation_pages.add(int(page_id))

    with open(out_json, 'w') as f:
        json.dump(list(disambiguation_pages), f)
