import requests
from qanta.datasets.quiz_bowl import QuestionDatabase

query = 'https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&titles={}'
answers = QuestionDatabase().all_answers().values()
for answer in answers:
    r =
    requests.get(query.format(answer))
    r = r.json()
