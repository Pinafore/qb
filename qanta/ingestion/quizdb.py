"""
This code is not directly called anywhere in the codebase, but contains the scripts
used to download QuizDB raw data dumps which are then hosted on AWS and processed
via qanta.ingestion.pipeline.
"""
import time
import os
import requests
from qanta import qlogging
from tqdm import tqdm


log = qlogging.get(__name__)

TOSSUP_URL = 'https://www.quizdb.org/admin/tossups.json?order=id_desc&page={page}&per_page=100'
TOURNAMENT_URL = 'https://www.quizdb.org/admin/tournaments.json?order=year_desc&page={page}&per_page=100'
CATEGORY_URL = 'https://www.quizdb.org/admin/categories.json?page={page}&per_page=100'
SUBCATEGORY_URL = 'https://www.quizdb.org/admin/subcategories.json?order=name_asc&page={page}&per_page=100'
BONUSES_URL = 'https://www.quizdb.org/admin/bonuses.json?order=id_desc&page={page}&per_page=100'
QUIZ_DB_SESSION = os.environ.get('QUIZ_DB_SESSION')


def fetch_authenticated_url(url):
    cookie = {'_quizdb_session': QUIZ_DB_SESSION}
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
    delay = 2
    should_sleep = False
    all_resources = []
    for page in tqdm(range(start_page, end_page)):
        log.info(f'Fetching page: {page}')
        resources, response_time = fetch_page_function(page)
        all_resources.extend(resources)
        log.info(f'Found {len(resources)} questions in {response_time}s')

        if len(resources) == 0:
            log.info(f'No resources found on page: {page}, exiting')
            break

        if response_time > 1:
            log.info(f'Response time is {response_time}s, waiting for 10s')
            should_sleep = True
        elif response_time > .5 and delay != 4:
            log.info(f'Response time is {response_time}s, increasing delay to 4s')
            delay = 4
        elif delay != 2:
            log.info(f'Response time is {response_time}s, decreasing delay to 2s')
            delay = 2

        if should_sleep:
            should_sleep = False
            time.sleep(10)
        else:
            time.sleep(delay)
    return all_resources


def fetch_all_tossups(start_page, end_page):
    return fetch_paginated_resource(fetch_tossup_page, start_page, end_page)


def fetch_all_bonuses(start_page, end_page):
    return fetch_paginated_resource(fetch_bonuses_page, start_page, end_page)
