import json
from qanta.ingestion.preprocess import format_qanta_json, add_sentences_

DIRECT_MAP = {}

# Mapping for PACE 2018
# DIRECT_MAP = {
    # 2: ('syracuse', 'Syracuse,_Sicily'),
    # 40: ('fyodor_dostoyevsky', 'Fyodor_Dostoevsky'),
    # 80: ('james_maxwell', 'James_Clerk_Maxwell'),
    # 106: ('syracuse', 'Syracuse,_Sicily'),
    # 131: ('the_dead', 'The_Dead_(short_story)'),
    # 144: ('if', None),
    # 169: ('the_stranger', 'The_Stranger_(novel)'),
    # 171: ('the_stranger', 'The_Stranger_(novel)'),
    # 172: ('the_stranger', 'The_Stranger_(novel)'),
    # 173: ('the_stranger', 'The_Stranger_(novel)'),
    # 188: ('uber_(company)', 'Uber'),
    # 205: ('gone_with_the_wind', None),
    # 207: ('liberal_democratic_party', None),
    # 215: ('readymades', None),
    # 272: ('gel_permeation_chromatography', None),
    # 279: ('nuclear_membrane', 'Nuclear_envelope'),
    # 291: ('dienes', None),
    # 294: ('nitro', 'Nitro_compound'),
    # 310: ('folic_acid', 'Folate'),
    # 314: ('polarity', 'Chemical_polarity'),
    # 317: ('ring', None),
    # 360: ('the_president', 'President_of_the_United_States'),
    # 373: ('rsa', 'RSA_(cryptosystem)'),
    # 390: ('charle', None),
    # 400: ('hermann', 'Hermann_Göring'),
    # 417: ('abc', None),
    # 449: ('multiplier', 'Multiplication'),
    # 454: ('relativity', None),
    # 455: ('buffalo', 'Buffalo,_New_York'),
    # 463: ('magyar', None),
    # 501: ('tcp', 'Transmission_Control_Protocol'),
    # 512: ('gone_with_the_wind', None),
    # 523: ('doom', 'Doom_(1993_video_game)'),
    # 539: ('christ_the_redeemer', 'Christ_the_Redeemer_(statue)'),
    # 543: ('painted_desert', None),
    # 549: ('great_schism', 'East–West_Schism'),
    # 576: ('japanese_internment', 'Internment_of_Japanese_Americans'),
    # 578: ('mather', 'Mather_(surname)'),
    # 579: ('tommy', 'Tommy_(album)'),
    # 583: ('gone_girl', None),
    # 591: ('the_kiss', None),
    # 606: ('the_fight_club', 'Fight_Club'),
    # 668: ('bowling_ball', 'Bowling_Balls'),
    # 669: ('bowling_ball', 'Bowling_Balls'),
    # 690: ('hemlock', 'Conium_maculatum'),
    # 692: ('tcp', 'Transmission_Control_Protocol'),
    # 695: ('angels_in_america:_a_gay_fantasia_on_national_themes', 'Angels_in_America'),
    # 696: ('angels_in_america:_a_gay_fantasia_on_national_themes', 'Angels_in_America'),
    # 693: ('hemlock', 'Conium_maculatum'),
    # 734: ('jackson', None),
    # 757: ('jackson', None),
    # 767: ('centaurs', 'Centaur'),
    # 768: ('angels_in_america:_a_gay_fantasia_on_national_themes', 'Angels_in_America'),
    # 772: ('bowling_ball', 'Bowling_Balls'),
    # 774: ('bowling_ball', 'Bowling_Balls')
# }


class Trickme:
    @staticmethod
    def parse_tossups(qanta_ds_path='data/external/datasets/qanta.mapped.2018.04.18.json',
                      trick_path='data/external/datasets/trickme_questions_12-15-2018.json',
                      start_idx=2000000, version='2018.04.18'):
        with open(qanta_ds_path) as f:
            qanta_ds = json.load(f)['questions']
        answer_set = {q['page'] for q in qanta_ds if q['page'] is not None}
        lookup = {a.lower().replace(' ', '_'): a for a in answer_set}
        with open(trick_path) as f:
            questions = []
            for i, q in enumerate(json.load(f)):
                if 'Question' in q:
                    text = q['Question']
                elif 'question' in q:
                    text = q['question']
                else:
                    raise ValueError('Could not find question field in question')

                if 'Answer' in q:
                    answer = q['Answer'].replace(' ', '_')
                elif 'answer' in q:
                    answer = q['answer'].replace(' ', '_')
                else:
                    raise ValueError('Could not find answer field in question')
                if len(answer) == 0 or len(text) == 0:
                    continue
                if i in DIRECT_MAP:
                    m_ans, m_page = DIRECT_MAP[i]
                    if m_ans == answer:
                        if m_page is None:
                            continue  # Skip this explicitly
                        elif m_page in answer_set:
                            page = m_page
                        else:
                            raise ValueError(f'{m_page} not in answer set\n Q: {text}')
                    else:
                        raise ValueError(f'Mapping error: {answer} != {m_ans}')
                elif answer in lookup:
                    page = lookup[answer]
                else:
                    raise ValueError(f'Could not find: idx: {i} Q:"{text}" \nA: "{answer}"')
                q_out = {
                    'text': text,
                    'answer': answer,
                    'page': page,
                    'fold': 'advtest',
                    'year': 2018,
                    'dataset': 'trickme',
                    'proto_id': None,
                    'qdb_id': None,
                    'trickme_id': i,
                    'difficulty': None,
                    'category': None,
                    'subcategory': None,
                    'qanta_id': start_idx + i,
                    'tournament': 'Adversarial Question Writing UMD December 15',
                    'gameplay': False
                }
                if 'email' in q:
                    q_out['author_email'] = q['email']
                if 'category' in q and q['category'] != "None":
                    q_out['category'] = q['category']
                questions.append(q_out)
            add_sentences_(questions, parallel=True)
            dataset = format_qanta_json(questions, version)
            return dataset

