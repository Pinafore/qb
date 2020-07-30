import json
import typer
import pandas as pd


app = typer.Typer()


@app.command()
def protobowl_to_feather(
    proto_path: str = "data/external/datasets/protobowl/protobowl-042818.log",
    out_path: str = "data/external/datasets/protobowl/protobowl-042818.feather",
):
    rows = []
    with open(proto_path) as f:
        record = json.loads(f)["object"]
        rows.append(
            {
                "user_id": record["id"],
                "proto_id": record["qid"],
                "time_elapsed": record["time_elapsed"],
                "time_remaining": record["time_remaining"],
                "ruling": record["ruling"],
                "answer": record["answer"],
                "guess": record["guess"],
                "question_text": record["question_text"],
                "year": record["question_info"]["year"],
                "tournament": record["question_info"]["tournament"],
                "category": record["question_info"]["category"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_feather(out_path)


if __name__ == "__main__":
    app()
