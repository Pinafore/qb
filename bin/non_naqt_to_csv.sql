select question as qnum, sent, raw, page, fold
from text t
left join questions q on t.question = q.id
where page != '' and (fold = 'guesstrain' or fold = 'guessdev' or fold = 'test')