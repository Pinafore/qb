import json

DIRECT_MAP = {
    2: ('syracuse', 'Syracuse,_Sicily'),
    80: ('james_maxwell', 'James_Clerk_Maxwell'),
    106: ('syracuse', 'Syracuse,_Sicily'),
    131: ('the_dead', 'The_Dead_(short_story)'),
    144: (None, None)
}


class Trickme:
    @staticmethod
    def parse_tossups(qanta_ds_path='data/external/datasets/qanta.mapped.2018.04.18.json',
                      trick_path='data/external/datasets/trickme_questions.json'):
        with open(qanta_ds_path) as f:
            qanta_ds = json.load(f)['questions']
        answer_set = {q['page'] for q in qanta_ds if q['page'] is not None}
        lookup = {a.lower().replace(' ', '_'): a for a in answer_set}
        with open(trick_path) as f:
            questions = []
            for i, q in enumerate(json.load(f)):
                text = q['Question']
                answer = q['Answer'].replace(' ', '_')
                if i in DIRECT_MAP:
                    m_ans, m_page = DIRECT_MAP[i]
                    if m_page is None:
                        continue # Skip this explicitly
                    elif m_ans == answer:
                        if m_page in answer_set:
                            page = m_page
                        else:
                            raise ValueError(f'{m_page} not in answer set')
                    else:
                        raise ValueError(f'Mapping error: {answer} != {m_ans}')
                elif answer in lookup:
                    page = lookup[answer]
                else:
                    raise ValueError(f'Could not find: idx: {i} Q:"{text}" \nA: "{answer}"')
                questions.append({
                    'text': text,
                    'answer': answer, 'page': page,
                    'fold': 'advtest',
                    'year': 2018,
                    'dataset': 'trickme',
                    'proto_id': None,
                    'qdb_id': None,
                    'difficulty': None,
                    'category': None,
                    'subcategory': None
                })
            return questions

