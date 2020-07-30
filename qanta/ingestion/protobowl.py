from qanta.spark import create_spark_session


def compute_question_player_counts(proto_log_path):
    spark = create_spark_session()
    df = spark.read.json(proto_log_path)
    df.createOrReplaceTempView("logs")
    question_player_counts = spark.sql(
        """
        SELECT object.qid, size(collect_set(object.user.id)) AS n_players
        FROM logs
        GROUP BY object.qid
    """
    ).collect()
    return {r.qid: r.n_players for r in question_player_counts}
