import math
import os
import pickle
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import typer
from pedroai.io import read_json, write_json
from pedroai.plot import theme_pedroai
from rich.console import Console
from rich.progress import track

from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.reporting.curve_score import CurveScore
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


@app.command()
def plot_empirical_buzz():
    proto_df = pd.read_hdf("data/external/datasets/protobowl/protobowl-042818.log.h5")
    dataset = read_json(QANTA_MAPPED_DATASET_PATH)
    questions = {q["qanta_id"]: q for q in dataset["questions"]}
    proto_to_question = {q["proto_id"]: q for q in dataset["questions"]}
    folds = {
        q["proto_id"]: q["fold"]
        for q in questions.values()
        if q["proto_id"] is not None
    }
    proto_df["fold"] = proto_df["qid"].map(lambda x: folds[x] if x in folds else None)
    proto_df["n"] = 1
    buzztest_df = proto_df[proto_df.fold == "buzztest"]
    play_counts = (
        buzztest_df.groupby("qid")
        .count()
        .reset_index()
        .sort_values("fold", ascending=False)
    )
    qid_to_counts = {r.qid: r.n for r in play_counts.itertuples()}
    popular_questions = play_counts.qid.tolist()
    curve = CurveScore()
    x = np.linspace(0, 1, 100)
    y = [curve.get_weight(n) for n in x]
    curve_df = pd.DataFrame({"buzzing_position": x, "result": y})
    curve_df["qid"] = "Expected Wins Curve Score"
    curve_df["source"] = "Curve Score | Average"
    proto_ids = popular_questions[:10]
    frames = []
    for proto_id in proto_ids:
        plays = buzztest_df[buzztest_df.qid == proto_id].sort_values("buzzing_position")
        plays = plays[plays.result != "prompt"]
        plays["result"] = plays["result"].astype(int)
        frames.append(plays)
    sample_df = pd.concat(frames)

    rows = []
    for qid, group_df in sample_df.groupby("qid"):
        n_opp_correct = 0
        n_opp_total = 0
        n = qid_to_counts[qid]
        rows.append(
            {
                "buzzing_position": 0,
                "n_opp_correct": 0,
                "n_opp_total": 1,
                "qid": f"Question with {n} Plays",
                "source": "Single Question",
                "n_plays": n,
            }
        )
        for r in group_df.itertuples():
            if r.result == 1:
                n_opp_correct += 1
            n_opp_total += 1
            rows.append(
                {
                    "buzzing_position": r.buzzing_position,
                    "n_opp_correct": n_opp_correct,
                    "n_opp_total": n_opp_total,
                    "qid": f"Question with {n} Plays",
                    "source": "Single Question",
                    "n_plays": n,
                }
            )
    n_opp_correct = 0
    n_opp_total = 0
    for r in sample_df.sort_values("buzzing_position").itertuples():
        if r.result == 1:
            n_opp_correct += 1
        n_opp_total += 1
        rows.append(
            {
                "buzzing_position": r.buzzing_position,
                "n_opp_correct": n_opp_correct,
                "n_opp_total": n_opp_total,
                "qid": "Average of Most Played",
                "source": "Curve Score | Average",
            }
        )

    df = pd.DataFrame(rows)
    df["p_opp_correct"] = df["n_opp_correct"] / df["n_opp_total"]
    df["p_win"] = 1 - df["p_opp_correct"]
    df["result"] = df["p_win"]

    def order(c):
        if c.startswith("Expected"):
            return -1000
        elif c.startswith("Average"):
            return -999
        elif c.startswith("Question with"):
            return -int(c.split()[2])
        else:
            return 1000

    categories = list(set(df.qid.tolist()) | set(curve_df.qid.tolist()))
    categories = sorted(categories, key=order)
    categories = pd.CategoricalDtype(categories, ordered=True)
    df["qid"] = df["qid"].astype(categories)
    cmap = plt.get_cmap("tab20")
    colors = [matplotlib.colors.to_hex(c) for c in cmap.colors]
    chart = (
        p9.ggplot(
            df[df.n_opp_total > 4],
            p9.aes(x="buzzing_position", y="result", color="qid"),
        )
        + p9.geom_line(p9.aes(linetype="source"))
        + p9.geom_line(
            p9.aes(x="buzzing_position", y="result", linetype="source"), data=curve_df
        )
        + p9.labs(
            x="Position in Question (%)",
            y="Empirical Probability of Winning",
            linetype="Data Type",
            color="Data Source",
        )
        + p9.scale_color_manual(values=colors)
        + theme_pedroai()
        + p9.theme(legend_position="right")
    )
    chart.save("output/empirical_buzz.pdf")


if __name__ == "__main__":
    app()
