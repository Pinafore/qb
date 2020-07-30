from typing import List
from collections import Counter

import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from qanta.datasets.quiz_bowl import QuestionDatabase, Question


def compute_stats(questions: List[Question], db_path):
    n_total = len(questions)
    n_guesser_train = sum(1 for q in questions if q.fold == "guesstrain")
    n_guesser_dev = sum(1 for q in questions if q.fold == "guessdev")
    n_buzzer_train = sum(1 for q in questions if q.fold == "buzzertrain")
    n_buzzer_dev = sum(1 for q in questions if q.fold == "buzzerdev")
    n_dev = sum(1 for q in questions if q.fold == "dev")
    n_test = sum(1 for q in questions if q.fold == "test")
    columns = [
        "N Total",
        "N Guesser Train",
        "N Guesser Dev",
        "N Buzzer Train",
        "N Buzzer Dev",
        "N Dev",
        "N Test",
    ]
    data = np.array(
        [
            n_total,
            n_guesser_train,
            n_guesser_dev,
            n_buzzer_train,
            n_buzzer_dev,
            n_dev,
            n_test,
        ]
    )
    norm_data = 100 * data / n_total

    return html.Div(
        [
            html.Label("Database Path"),
            html.Div(db_path),
            html.H2("Fold Distribution"),
            html.Table(
                [
                    html.Tr([html.Th(c) for c in columns]),
                    html.Tr([html.Td(c) for c in data]),
                    html.Tr([html.Td(f"{c:.2f}%") for c in norm_data]),
                ]
            ),
        ]
    )


def display_question(question: Question):
    return html.Div([html.Label("ID"), html.Span(question.qnum)])


def main():
    db = QuestionDatabase()
    question_lookup = db.all_questions()
    questions = list(question_lookup.values())

    guesser_train_questions = [q for q in questions if q.fold == "guesstrain"]
    guesser_train_answers = [q.page for q in guesser_train_questions]
    answer_counts = Counter(guesser_train_answers)
    answer_set = set(answer_counts.keys())

    app = dash.Dash()
    app.layout = html.Div(
        children=[
            html.H1(children="Quiz Bowl Question Explorer"),
            compute_stats(questions, db.location),
            html.H2("Question Inspector"),
            dcc.Dropdown(
                options=[{"label": q.qnum, "value": q.qnum} for q in questions],
                value=questions[0].qnum,
                id="question-selector",
            ),
            html.Div([html.Div(id="question-display")]),
            dcc.Graph(
                id="answer-count-plot",
                figure=go.Figure(
                    data=[
                        go.Histogram(
                            x=list(answer_counts.values()), name="Answer Counts"
                        )
                    ],
                    layout=go.Layout(
                        title="Answer Count Distribution", showlegend=True
                    ),
                ),
            ),
            dcc.Graph(
                id="answer-count-cum-plot",
                figure=go.Figure(
                    data=[
                        go.Histogram(
                            x=list(answer_counts.values()),
                            name="Answer Counts Cumulative",
                            cumulative=dict(enabled=True, direction="decreasing"),
                            histnorm="percent",
                        )
                    ],
                    layout=go.Layout(
                        title="Answer Count Cumulative Distribution", showlegend=True
                    ),
                ),
            ),
            html.Label("Answer Selection"),
            dcc.Dropdown(
                options=sorted(
                    [{"label": a, "value": a} for a in answer_set],
                    key=lambda k: k["label"],
                ),
                id="answer-list",
            ),
            html.Div(id="answer-count"),
        ]
    )

    @app.callback(
        Output(component_id="answer-count", component_property="children"),
        [Input(component_id="answer-list", component_property="value")],
    )
    def update_answer_count(answer):
        return f"Answer: {answer} Question Count: {answer_counts[answer]}"

    @app.callback(
        Output(component_id="question-display", component_property="children"),
        [Input(component_id="question-selector", component_property="value")],
    )
    def update_question(qb_id):
        qb_id = int(qb_id)
        question = question_lookup[qb_id]
        sentences, answer, _ = question.to_example()
        return (
            [html.P(f"ID: {qb_id} Fold: {question.fold}"), html.H3("Sentences")]
            + [html.P(f"{i}: {sent}") for i, sent in enumerate(sentences)]
            + [html.H3("Answer"), html.P(answer)]
        )

    app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
