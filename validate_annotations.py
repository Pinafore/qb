from qanta import qlogging
from qanta.ingestion.answer_mapping import read_wiki_titles
from qanta.ingestion.annotated_mapping import PageAssigner


log = qlogging.get("validate_annotations")


def normalize(title):
    return title.replace(" ", "_")


def check_page(page, titles):
    n_page = normalize(page)
    if n_page not in titles:
        log.error(f"Title not found: {page}")


def main():
    titles = read_wiki_titles()
    assigner = PageAssigner()
    log.info("Checking direct protobowl mappings...")
    for page in assigner.protobowl_direct.values():
        check_page(page, titles)

    log.info("Checking direct quizdb mappings...")
    for page in assigner.quizdb_direct.values():
        check_page(page, titles)

    log.info("Checking unambiguous mappings...")
    for page in assigner.unambiguous.values():
        check_page(page, titles)

    log.info("Checking ambiguous mappings...")
    for entry in assigner.ambiguous.values():
        for option in entry:
            check_page(option["page"], titles)


if __name__ == "__main__":
    main()
