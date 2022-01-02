#!/usr/bin/env python
"""
CLI utilities for QANTA
"""

from typing import Dict, Optional
import random
import sqlite3
import csv
from collections import defaultdict
import json
from os import path
import click
import yaml
from jinja2 import Environment, PackageLoader
import tqdm

from qanta import qlogging
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.elasticsearch import elasticsearch_cli
from qanta.util.environment import ENVIRONMENT
from qanta.util.io import safe_open, shell, get_tmp_filename
from qanta.util.constants import QANTA_SQL_DATASET_PATH, GUESSER_GENERATION_FOLDS
from qanta.hyperparam import expand_config
from qanta.wikipedia.categories import categorylinks_cli
from qanta.wikipedia.vital import vital_cli
from qanta.ingestion.trickme import trick_cli
from qanta.ingestion.command import ingestion_cli

log = qlogging.get("cli")

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


main.add_command(categorylinks_cli, name="categories")
main.add_command(vital_cli, name="vital")
main.add_command(elasticsearch_cli, name="elasticsearch")
main.add_command(trick_cli, name="trick")
main.add_command(ingestion_cli, name="map")


@main.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=5000)
@click.option("--debug", default=False)
@click.argument("guessers", nargs=-1)
def guesser_api(host, port, debug, guessers):
    if debug:
        log.warning(
            "WARNING: debug mode can expose environment variables (AWS keys), NEVER use when API is exposed to  web"
        )
        log.warning("Confirm that you would like to enable flask debugging")
        confirmation = input("yes/no:\n").strip()
        if confirmation != "yes":
            raise ValueError("Most confirm enabling debug mode")

    AbstractGuesser.multi_guesser_web_api(guessers, host=host, port=port, debug=debug)


def run_guesser(n_times, workers, guesser_qualified_class):
    for _ in range(n_times):
        if "qanta.guesser" not in guesser_qualified_class:
            log.error(
                "qanta.guesser not found in guesser_qualified_class, this is likely an error, exiting."
            )
            return
        shell("rm -rf /tmp/qanta")
        shell(f"rm -rf output/guesser/{guesser_qualified_class}")
        shell(
            f"luigi --local-scheduler --module qanta.pipeline.guesser --workers {workers} AllSingleGuesserReports"
        )


@main.command()
@click.option("--n_times", default=1)
@click.option("--workers", default=1)
@click.argument("guesser_qualified_class")
def guesser_pipeline(n_times, workers, guesser_qualified_class):
    run_guesser(n_times, workers, guesser_qualified_class)


@main.command()
@click.option("--n", default=20)
@click.option("--seed", default=0)
def sample_answer_pages(n, seed):
    """
    Take a random sample of n questions, then return their answers and pages
    formatted for latex in the journal paper
    """
    with open("data/external/datasets/qanta.mapped.2021.12.20.json") as f:
        questions = json.load(f)["questions"]
        random.seed(seed)
        random.shuffle(questions)
    for i, q in enumerate(questions[:n]):
        answer = q["answer"]
        page = q["page"]
        if i - 1 == n:
            latex_format = r"{answer} & {page}\\ \midrule"
        else:
            latex_format = r"{answer} & {page}\\ \bottomrule"
        answer = answer.replace("{", r"\{").replace("}", r"\}").replace("_", r"\_")
        if page is None:
            page = r"\textbf{No Mapping Found}"
        else:
            page = page.replace("{", r"\{").replace("}", r"\}").replace("_", r"\_")
        print(latex_format.format(answer=answer, page=page))


@main.command()
@click.argument("base_file")
@click.argument("hyper_file")
@click.argument("output_file")
def hyper_to_conf(base_file, hyper_file, output_file):
    expand_config(base_file, hyper_file, output_file)


def get_slurm_config_value(
    name: str, default_config: Dict, guesser_config: Optional[Dict]
):
    if guesser_config is None:
        return default_config[name]
    else:
        if name in guesser_config:
            return guesser_config[name]
        else:
            return default_config[name]


@main.command()
@click.option("--slurm-config-file", default="slurm-config.yaml")
@click.argument("task")
@click.argument("output_dir")
def generate_guesser_slurm(slurm_config_file, task, output_dir):
    with open(slurm_config_file) as f:
        slurm_config = yaml.load(f)
        default_slurm_config = slurm_config["default"]
    env = Environment(loader=PackageLoader("qanta", "slurm/templates"))
    template = env.get_template("guesser-luigi-template.sh")
    enabled_guessers = list(AbstractGuesser.list_enabled_guessers())

    for i, gs in enumerate(enabled_guessers):
        if gs.guesser_class == "ElasticSearchGuesser":
            raise ValueError("ElasticSearchGuesser is not compatible with slurm")
        elif gs.guesser_class in slurm_config:
            guesser_slurm_config = slurm_config[gs.guesser_class]
        else:
            guesser_slurm_config = None
        partition = get_slurm_config_value(
            "partition", default_slurm_config, guesser_slurm_config
        )
        qos = get_slurm_config_value("qos", default_slurm_config, guesser_slurm_config)
        mem_per_cpu = get_slurm_config_value(
            "mem_per_cpu", default_slurm_config, guesser_slurm_config
        )
        gres = get_slurm_config_value(
            "gres", default_slurm_config, guesser_slurm_config
        )
        max_time = get_slurm_config_value(
            "max_time", default_slurm_config, guesser_slurm_config
        )
        cpus_per_task = get_slurm_config_value(
            "cpus_per_task", default_slurm_config, guesser_slurm_config
        )
        account = get_slurm_config_value(
            "account", default_slurm_config, guesser_slurm_config
        )
        if task == "GuesserReport":
            folds = GUESSER_GENERATION_FOLDS
        else:
            folds = []
        script = template.render(
            {
                "task": task,
                "guesser_module": gs.guesser_module,
                "guesser_class": gs.guesser_class,
                "dependency_module": gs.dependency_module,
                "dependency_class": gs.dependency_class,
                "config_num": gs.config_num,
                "partition": partition,
                "qos": qos,
                "mem_per_cpu": mem_per_cpu,
                "max_time": max_time,
                "gres": gres,
                "cpus_per_task": cpus_per_task,
                "account": account,
                "folds": folds,
            }
        )
        slurm_file = path.join(output_dir, f"slurm-{i}.sh")
        with safe_open(slurm_file, "w") as f:
            f.write(script)

    singleton_path = "qanta/slurm/templates/guesser-singleton.sh"
    singleton_output = path.join(output_dir, "guesser-singleton.sh")
    shell(f"cp {singleton_path} {singleton_output}")

    master_template = env.get_template("guesser-master-template.sh")
    master_script = master_template.render(
        {
            "script_list": [
                path.join(output_dir, f"slurm-{i}.sh")
                for i in range(len(enabled_guessers))
            ]
            + [singleton_output],
            "gres": gres,
            "partition": partition,
            "qos": qos,
            "mem_per_cpu": mem_per_cpu,
            "max_time": max_time,
            "gres": gres,
            "cpus_per_task": cpus_per_task,
            "account": account,
        }
    )
    with safe_open(path.join(output_dir, "slurm-master.sh"), "w") as f:
        f.write(master_script)


@main.command()
@click.option("--partition", default="dpart")
@click.option("--qos", default="batch")
@click.option("--mem-per-cpu", default="8g")
@click.option("--max-time", default="1-00:00:00")
@click.option("--nodelist", default=None)
@click.option("--cpus-per-task", default=None)
@click.argument("luigi_module")
@click.argument("luigi_task")
def slurm(
    partition,
    qos,
    mem_per_cpu,
    max_time,
    nodelist,
    cpus_per_task,
    luigi_module,
    luigi_task,
):
    env = Environment(loader=PackageLoader("qanta", "slurm/templates"))
    template = env.get_template("luigi-template.sh.jinja2")
    sbatch_script = template.render(
        {
            "luigi_module": luigi_module,
            "luigi_task": luigi_task,
            "partition": partition,
            "qos": qos,
            "mem_per_cpu": mem_per_cpu,
            "max_time": max_time,
            "nodelist": nodelist,
            "cpus_per_task": cpus_per_task,
        }
    )
    tmp_file = get_tmp_filename()
    with open(tmp_file, "w") as f:
        f.write(sbatch_script)
    shell(f"sbatch {tmp_file}")
    shell(f"rm -f {tmp_file}")


@main.command()
def answer_map_google_csvs():
    from qanta.ingestion.gspreadsheets import create_answer_mapping_csvs

    create_answer_mapping_csvs()


@main.command()
@click.argument("question_tsv")
def process_annotated_test(question_tsv):
    import pandas as pd

    df = pd.read_csv(question_tsv, delimiter="\t")
    proto_questions = df[df.qdb_id.isna()]
    qdb_questions = df[df.proto_id.isna()]
    qdb_map = {
        int(q.qdb_id): q.page for q in qdb_questions.itertuples() if type(q.page) is str
    }
    proto_map = {
        q.proto_id: q.page for q in proto_questions.itertuples() if type(q.page) is str
    }
    print("Proto lines")
    for qid, page in proto_map.items():
        print(f"  {qid}: {page}")

    print("QDB lines")
    for qid, page in qdb_map.items():
        print(f"  {qid}: {page}")

    print("Unmappable proto")
    for r in proto_questions.itertuples():
        if type(r.page) is not str:
            print(f"  - {r.proto_id}")

    print("Unmappable qdb")
    for r in qdb_questions.itertuples():
        if type(r.page) is not str:
            print(f"  - {int(r.qdb_id)}")


if __name__ == "__main__":
    main()
