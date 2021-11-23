import argparse
import json
from pathlib import Path

DS_VERSION = "2018.04.18"
LOCAL_QANTA_PREFIX = "data/external/datasets/"
QANTA_TRAIN_DATASET_PATH = f"qanta.train.{DS_VERSION}.json"
QANTA_DEV_DATASET_PATH = f"qanta.dev.{DS_VERSION}.json"
QANTA_TEST_DATASET_PATH = f"qanta.test.{DS_VERSION}.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for split, path in [('train', QANTA_TRAIN_DATASET_PATH), ('dev', QANTA_DEV_DATASET_PATH), ('test', QANTA_TEST_DATASET_PATH)]:
        with open(Path(LOCAL_QANTA_PREFIX) / path) as f:
            data = json.load(f)
        
        output = []
        for q in data['questions']:
            output.append({'uid': q['qanta_id'], 'question': q['text'], 'answer': q['page'], 'context': ''})
        

        with open(output_dir / f'qb-{split}-{DS_VERSION}.jsonl', 'w') as f:
            for r in output:
                f.write(f'{json.dumps(r)}\n')


if __name__ == '__main__':
    main()