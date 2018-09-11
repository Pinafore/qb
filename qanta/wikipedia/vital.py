"""
Code to fetch Wikipedia level 5 vital articles. This is a good representation
of the most important 50,000 Wikipedia articles. The actual number is closer to 30,000
but the target number Wikimedia has set to have is 50,000
"""

import urllib
import json
from bs4 import BeautifulSoup
import requests
import click


def fetch_vital_titles():
    vital_parent_html = requests.get(
        'https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5'
    ).content.decode()
    parent_soup = BeautifulSoup(vital_parent_html, 'html.parser')
    vital_links = []
    for link in parent_soup.find_all('a'):
        url = link.get('href')
        if url is not None and url.startswith('/wiki/Wikipedia:Vital_articles/Level/5/'):
            vital_links.append(url)

    vital_articles = set()
    for link in vital_links:
        soup = BeautifulSoup(requests.get(f'https://en.wikipedia.org/{link}').content, 'lxml')
        for page_link in soup.find_all('a'):
            url = page_link.get('href')
            if url is None:
                continue
            url = urllib.parse.unquote(url)
            if url is not None and url.startswith('/wiki') and ':' not in url:
                vital_articles.add(url.split('/')[2])
    return vital_articles


@click.group()
def vital_cli():
    pass


@vital_cli.command()
@click.argument('path')
def write(path):
    vital_articles = list(fetch_vital_titles())
    with open(path, 'w') as f:
        json.dump(vital_articles, f)
