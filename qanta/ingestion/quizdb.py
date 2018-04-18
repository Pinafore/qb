import time
import os
import requests
from qanta import qlogging
from tqdm import tqdm


log = qlogging.get(__name__)

TOSSUP_URL = 'https://www.quizdb.org/admin/tossups.json?order=id_desc&page={page}&per_page=100'


def fetch_page(login_cookie, page):
    cookie = {'_quizdb_session': login_cookie}
    start = time.time()
    r = requests.get(TOSSUP_URL.format(page=page), cookies=cookie)
    end = time.time()
    if r.ok:
        return r.json(), end - start
    else:
        raise ValueError(f'Encountered a bad response: {r.status_code} {r.text}')


def robust_fetch_page(login_cookie, page):
    try:
        return fetch_page(login_cookie, page)
    except Exception as e:
        log.info(f'Ran into exception, waiting for 30s then retrying one more time before exiting: {e}')
        time.sleep(30)
        return fetch_page(login_cookie, page)


def fetch_all_questions(start_page, end_page):
    quizdb_session = os.environ.get('QUIZ_DB_SESSION')
    if quizdb_session is None:
        raise ValueError('Cannot scrap quizdb.org since no authentication credentials found in QUIZ_DB_SESSION')

    delay = 2
    should_sleep = False
    all_questions = []
    for page in tqdm(range(start_page, end_page)):
        log.info(f'Fetching page: {page}')
        questions, response_time = robust_fetch_page(quizdb_session, page)
        all_questions.extend(questions)
        log.info(f'Found {len(questions)} questions in {response_time}s')

        if len(questions) == 0:
            log.info(f'No questions found on page: {page}, exiting')
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
    return all_questions
