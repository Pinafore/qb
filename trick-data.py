import json
from collections import Counter
import pandas as pd
import click


@click.group()
def main():
    pass


@main.command()
def validate():
    """
    This command validates that the dataset for trickme paper looks correct.
    It checks that:
    * The number of adversarial questions matches the number in the ID lookup file
    * ID lookup file is a map from qanta_id to source (ir vs rnn, round 1 etc)
    * The number of questions in dataset matches how many the experimental results have
    * That the qanta_id and page for each question match in dataset and experimental results
    """
    with open('data/external/datasets/qanta.expo.2018.04.18.json') as f:
        questions = [q for q in json.load(f)['questions']]
    with open('data/external/datasets/trickme-id-model.json') as f:
        id_to_model = json.load(f)
        id_to_model = {int(k): v for k, v in id_to_model.items()}

    print('Lengths should be 946')
    print(len(questions))
    print(len(id_to_model))
    print(Counter(id_to_model.values()))
    qid_to_page = {q['qanta_id']: q['page'] for q in questions}
    qids = {q['qanta_id'] for q in questions}
    id_to_model_qids = {k for k in id_to_model.keys()}
    print(len(qids.intersection(id_to_model_qids)))
    df = pd.read_json('output/tacl/all_rounds_df.json')
    # This ID range is used for trickme questions
    df = df[df.qanta_id >= 2_000_000][['qanta_id', 'page']]
    experiments_qid_to_page = {t.qanta_id: t.page for t in df.itertuples()}
    print(len(experiments_qid_to_page))
    for qid, page in experiments_qid_to_page.items():
        if qid_to_page[qid] != page:
            raise ValueError(f'exp: {qid} {page} data: {qid_to_page[qid]}')


@main.command()
def merge():
    """
    To create final dataset we take the expo/adversarial question file and append
    information about what interface was used to create it.
    """
    with open('data/external/datasets/trickme-id-model.json') as f:
        id_to_model = json.load(f)
        id_to_model = {int(k): v for k, v in id_to_model.items()}

    with open('data/external/datasets/qanta.expo.2018.04.18.json') as f:
        data = json.load(f)
        data['bibtex'] = (
            '@inproceedings{Wallace2019Trick,'
            '  title={Trick Me If You Can: Human-in-the-loop Generation of Adversarial Question Answering Examples},'
            '  author={Eric Wallace and Pedro Rodriguez and Shi Feng and Ikuya Yamada and Jordan Boyd-Graber},'
            '  booktitle = "Transactions of the Association for Computational Linguistics"'
            '  year={2019}, '
            '}'
        )
        data['date'] = '2019-04-30'
        data['project_website'] = 'http://trickme.qanta.org'
        data['dependent_checksums'] = {
            'qanta.expo.2018.04.18.json': 'c56a129b4d9c925187e2e58cc51c0b77',
            'trickme-id-model.json': 'cb0e26e5c9d1cada7b0b9cd0edb6c9e5'
        }

        questions = data['questions']
        for q in questions:
            q['fold'] = 'adversarial'
            source = id_to_model[q['qanta_id']]
            if source == 'es':
                ui = 'ir-r1'
            elif source == 'es-2':
                ui = 'ir-r2'
            elif source == 'rnn':
                ui = 'rnn'
            else:
                raise ValueError(f'Unrecognized source: {source}')
            q['interface'] = ui

    with open('data/external/datasets/qanta.tacl-trick.json', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
