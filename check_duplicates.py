from rapidfuzz import process, fuzz
import numpy as np
import json
import pickle
import typer


app = typer.Typer()



@app.command()
def similarity(output_file: str):
    with open('data/external/datasets/qanta.mapped.2018.04.18.json') as f:
        questions = [q['text'] for q in json.load(f)['questions']]
    
    similarity = process.cdist(questions, questions, scorer=fuzz.ratio, workers=-1)
    with open(output_file, 'wb') as f:
        pickle.dump(similarity, f)


@app.command()
def duplicates(similarity_file: str, output_file: str):
    with open(similarity_file, 'rb') as f:
        results = pickle.load(f)
    
    with open('data/external/datasets/qanta.mapped.2018.04.18.json') as f:
        questions = json.load(f)['questions']
    
    qid_to_question = {}
    for q in questions:
        qid_to_question[q['qanta_id']] = q
    
    id_to_question = {}
    for i, q in enumerate(questions):
        id_to_question[i] = q

    similar = np.where(results > 95)
    n = len(similar[0])
    duplicates = []
    for si in range(n):
        i = similar[0][si]
        j = similar[1][si]
        if i != j and i < j:
            i_question = id_to_question[i]
            j_question = id_to_question[j]
            if i_question['page'] is None or j_question['page'] is None:
                continue
            else:
                duplicates.append([i_question['qanta_id'], j_question['qanta_id']])
    with open(output_file, 'w') as f:
        json.dump(duplicates, f)


if __name__ == '__main__':
    app()