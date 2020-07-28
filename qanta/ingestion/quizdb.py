"""
This code is not directly called anywhere in the codebase, but contains the scripts
used to download QuizDB raw data dumps which are then hosted on AWS and processed
via qanta.ingestion.pipeline.

This code should be called from the command line, but requires additional configuration
to work correctly. Namely, the quizdb API requires setting up the cookies associated with
an admin account. Be sure to fill these in where they are empty below the CHANGEME comment
"""
import time
import json
import os
import requests
from qanta import qlogging
from tqdm import tqdm
import typer

app = typer.Typer()


log = qlogging.get(__name__)

TOSSUP_URL = 'https://www.quizdb.org/admin/tossups.json?order=id_desc&page={page}&per_page=1000'
TOURNAMENT_URL = 'https://www.quizdb.org/admin/tournaments.json?order=year_desc&page={page}&per_page=1000'
CATEGORY_URL = 'https://www.quizdb.org/admin/categories.json?page={page}&per_page=1000'
SUBCATEGORY_URL = 'https://www.quizdb.org/admin/subcategories.json?order=name_asc&page={page}&per_page=1000'
BONUSES_URL = 'https://www.quizdb.org/admin/bonuses.json?order=id_desc&page={page}&per_page=1000'
QUIZ_DB_SESSION = os.environ.get('QUIZ_DB_SESSION')


def fetch_authenticated_url(url):
    # CHANGEME
    cookie = {
        '_quizdb_session': '',
        '_remember_admin_user_token': '',
        '_session_id': ''
    }
    start = time.time()
    r = requests.get(url, cookies=cookie)
    end = time.time()
    if r.ok:
        return r.json(), end - start
    else:
        raise ValueError(f'Encountered a bad response: {r.status_code} {r.text}')


def robust_fetch_authenticated_url(url):
    if QUIZ_DB_SESSION is None:
        raise ValueError('Cannot scrap quizdb.org since no authentication credentials found in QUIZ_DB_SESSION')
    try:
        return fetch_authenticated_url(url)
    except Exception as e:
        log.info(f'Ran into exception, waiting for 30s then retrying one more time before exiting: {e}')
        time.sleep(30)
        return fetch_authenticated_url(url)


def fetch_tossup_page(page):
    url = TOSSUP_URL.format(page=page)
    return robust_fetch_authenticated_url(url)


def fetch_tournament_page(page):
    url = TOURNAMENT_URL.format(page=page)
    return robust_fetch_authenticated_url(url)


def fetch_category_page(page):
    url = CATEGORY_URL.format(page=page)
    return robust_fetch_authenticated_url(url)


def fetch_subcategory_page(page):
    url = SUBCATEGORY_URL.format(page=page)
    return robust_fetch_authenticated_url(url)


def fetch_bonuses_page(page):
    url = BONUSES_URL.format(page=page)
    return robust_fetch_authenticated_url(url)


def fetch_paginated_resource(fetch_page_function, start_page, end_page):
    delay = .5
    all_resources = []
    for page in tqdm(range(start_page, end_page)):
        log.info(f'Fetching page: {page}')
        resources, response_time = fetch_page_function(page)
        all_resources.extend(resources)
        log.info(f'Found {len(resources)} items in {response_time}s')

        if len(resources) == 0:
            log.info(f'No resources found on page: {page}, exiting')
            break
        time.sleep(delay)
    return all_resources


def fetch_all_tossups(start_page, end_page):
    return fetch_paginated_resource(fetch_tossup_page, start_page, end_page)


def fetch_all_bonuses(start_page, end_page):
    return fetch_paginated_resource(fetch_bonuses_page, start_page, end_page)


@app.command()
def get_tossups(out_path: str):
    tossups = fetch_all_tossups(1, 1000)
    with open(out_path, 'w') as f:
        json.dump({'tossups': tossups}, f)


@app.command()
def get_bonuses(out_path: str):
    bonuses = fetch_all_bonuses(1, 1000)
    with open(out_path, 'w') as f:
        json.dump({'bonuses': bonuses}, f)


@app.command()
def get_tournaments(out_path: str):
    tournaments = fetch_paginated_resource(fetch_tournament_page, 1, 1000)
    with open(out_path, 'w') as f:
        json.dump({'tournaments': tournaments}, f)


@app.command()
def get_categories(out_path: str):
    categories = fetch_paginated_resource(fetch_category_page, 1, 1000)
    with open(out_path, 'w') as f:
        json.dump({'categories': categories}, f)

        
@app.command()
def get_subcategories(out_path: str):
    subcategories = fetch_paginated_resource(fetch_subcategory_page, 1, 1000)
    with open(out_path, 'w') as f:
        json.dump({'subcategories': subcategories}, f)

if __name__ == "__main__":
    app()