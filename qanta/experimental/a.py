import re
import wikipedia
from tqdm import tqdm
from qanta.bonus.dataset import BonusQuestion, BonusQuestionDatabase

def normalize_answer(answer):
    answer = re.sub('\(.*\)', '', answer)
    answer = re.sub('\[.*\]', '', answer)
    answer = answer.replace('_', ' ')
    answer = re.sub('\s+', ' ', answer).strip()
    return answer

def normalize_page_title(title):
    title = title.split()
    title = '_'.join(title)
    return title

qs = list(BonusQuestionDatabase().all_questions().values())
all_pages = {}
for q in tqdm(qs):
    for i in range(len(q.answers)):
        a = normalize_answer(q.answers[i])
        page = wikipedia.page(a)
        q.pages[i] = page.title
