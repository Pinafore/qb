from qanta.ingestion.answer_mapping import read_wiki_titles
from qanta.ingestion.annotated_mapping import PageAssigner


def normalize(title):
    return title.replace(' ', '_')


def main():
    titles = read_wiki_titles()
    assigner = PageAssigner()
    print('Checking direct protobowl mappings...')
    for page in assigner.protobowl_direct.values():
        page = normalize(page)
        if page not in titles:
            print(f'Title not found: {page}')

    print('Checking direct quizdb mappings...')
    for page in assigner.quizdb_direct.values():
        page = normalize(page)
        if page not in titles:
            print(f'Title not found: {page}')

    print('Checking unambiguous mappings...')
    for page in assigner.unambiguous.values():
        page = normalize(page)
        if page not in titles:
            print(f'Title not found: {page}')

    print('Checking ambiguous mappings...')
    for entry in assigner.ambiguous.values():
        for option in entry:
            page = normalize(option['page'])
            if page not in titles:
                print(f'Title not found: {page}')
if __name__ == '__main__':
    main()
