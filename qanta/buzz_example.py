import math
import os
import pickle
from collections import defaultdict

import pandas as pd
import typer
from pedroai.io import read_json, write_json
from rich.console import Console
from rich.progress import track

from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.util.constants import QANTA_MAPPED_DATASET_PATH

console = Console()
app = typer.Typer()


@app.command()
def export(output_file: str, fold: str = "buzztest"):
    fold = "buzztest"
    guesses_dir = AbstractGuesser.output_path("qanta.guesser.rnn", "RnnGuesser", 0, "")
    guesses_dir = AbstractGuesser.guess_path(guesses_dir, fold, "char")
    with open(guesses_dir, "rb") as f:
        guesses = pickle.load(f)
    guesses = guesses.groupby("qanta_id")

    questions = QuizBowlDataset(buzzer_train=True).questions_by_fold()
    questions = {q.qanta_id: q for q in questions[fold]}
    buzzers = {}
    for name in ["RNNBuzzer", "ThresholdBuzzer", "MLPBuzzer"]:
        model_dir = f"output/buzzer/{name}"
        buzzes_dir = os.path.join(model_dir, "{}_buzzes.pkl".format(fold))
        with open(buzzes_dir, "rb") as f:
            buzzers[name] = pickle.load(f)

    qid_to_buzzes = defaultdict(dict)
    for name, buzzes in track(buzzers.items()):
        for qid, (char_indices, scores) in buzzes.items():
            gs = (
                guesses.get_group(qid)
                .groupby("char_index")
                .aggregate(lambda x: x.head(1))
                .to_dict()["guess"]
            )
            question = questions[qid]
            q_len = len(question.text)
            buzz_oracle_position = -1
            buzz_model_position = -1
            oracle_guess = None
            buzz_guess = None
            for i, char_index in enumerate(char_indices):
                buzz_oracle = gs[char_index] == question.page
                if buzz_oracle:
                    if buzz_oracle_position == -1 or char_index <= buzz_oracle_position:
                        oracle_guess = question.page
                        buzz_oracle_position = char_index

                if scores[i][1] > scores[i][0]:
                    if buzz_model_position == -1 or char_index < buzz_model_position:
                        buzz_guess = gs[char_index]
                        buzz_model_position = char_index
            qid_to_buzzes[qid][name] = {
                "oracle": buzz_oracle_position,
                "oracle_fraction": buzz_oracle_position / q_len
                if buzz_oracle_position != -1
                else -1,
                "position": buzz_model_position,
                "position_fraction": buzz_model_position / q_len
                if buzz_model_position != -1
                else -1,
                "q_len": q_len,
                "oracle_guess": oracle_guess,
                "buzz_guess": buzz_guess,
                "answer": question.page,
                "impossible": oracle_guess is None,
            }
    write_json(output_file, qid_to_buzzes)


HUMAN = r"\tdiamond{{{}}}"
RNN = r"\tcircle{{{}}}"
MLP = r"\tsquare{{{}}}"
THRESHOLD = r"\ttriangle{{{}}}"

CORRECT = "cgreen"
WRONG = "cred"

NAME_TO_SYMBOL = {
    "human": HUMAN,
    "oracle": "*",
    "RNNBuzzer": RNN,
    "MLPBuzzer": MLP,
    "ThresholdBuzzer": THRESHOLD,
}

TEMPLATE = r"""
%s\\
\textbf{Answer:} \underline{%s}
"""


@app.command()
def latex(qid: int, buzz_file: str, output_file: str):
    questions = {
        q["qanta_id"]: q for q in read_json(QANTA_MAPPED_DATASET_PATH)["questions"]
    }
    buzzes = read_json(buzz_file)
    proto_df = pd.read_hdf("data/external/datasets/protobowl/protobowl-042818.log.h5")

    computer_buzzes = buzzes[str(qid)]
    proto_id = questions[qid]["proto_id"]
    human_buzzes = proto_df[proto_df.qid == proto_id]

    answer = questions[qid]["page"]
    question_buzzes = [
        {
            "name": "oracle",
            "fraction": computer_buzzes["RNNBuzzer"]["oracle_fraction"],
            "answer": answer,
            "guess": computer_buzzes["RNNBuzzer"]["oracle_guess"],
            "correct": answer == computer_buzzes["RNNBuzzer"]["oracle_guess"],
        }
    ]
    for name, computer_buzz in buzzes[str(qid)].items():
        question_buzzes.append(
            {
                "name": name,
                "fraction": computer_buzz["position_fraction"],
                "answer": answer,
                "guess": computer_buzz["buzz_guess"],
                "correct": computer_buzz["answer"] == computer_buzz["buzz_guess"],
            }
        )

    for row in human_buzzes.itertuples():
        question_buzzes.append(
            {
                "name": "human",
                "fraction": row.buzzing_position,
                "answer": answer,
                "guess": row.guess,
                "correct": row.result,
            }
        )
    tex_df = pd.DataFrame(question_buzzes)
    text = questions[qid]["text"]
    q_len = len(text)
    tex_df["char"] = tex_df["fraction"].map(lambda f: math.floor(f * q_len))
    char_to_symbols = {}
    for row in tex_df.itertuples():
        shape = NAME_TO_SYMBOL[row.name]
        position = row.char
        if row.correct:
            color = CORRECT
        else:
            color = WRONG
        colored_shape = shape.format(color)
        if position in char_to_symbols:
            char_to_symbols[position].append(colored_shape)
        else:
            char_to_symbols[position] = [colored_shape]

    characters = list(text)
    out_chars = []
    for idx in range(len(characters) - 1, -1, -1):
        out_chars.append(characters[idx])
        if idx in char_to_symbols:
            for symbol in char_to_symbols[idx]:
                out_chars.append(symbol)
    out_chars.reverse()
    out_text = "".join(out_chars)

    tex_out = TEMPLATE % (out_text, answer.replace("_", " "))
    console.log(buzzes[str(qid)])
    console.log(
        human_buzzes.drop(columns=["date", "qid"]).sort_values("buzzing_position")
    )

    with open(output_file, "w") as f:
        f.write(tex_out)


if __name__ == "__main__":
    app()
