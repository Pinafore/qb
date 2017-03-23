import json
import pickle
from pyspark import SparkConf, SparkContext, RDD, Broadcast
from qanta.util.environment import QB_SPARK_MASTER


def extract_property_map(parsed_wikidata: RDD):
    def parse_property(prop):
        label = prop['labels']['en']['value']
        return prop['id'], label
    return parsed_wikidata\
        .filter(lambda d: d['type'] == 'property')\
        .map(parse_property)\
        .collectAsMap()


def extract_item_page_map(wikidata_items: RDD):
    def parse_item_page(item):
        item_id = item['id']
        if 'enwiki' in item['sitelinks']:
            return [(item_id, item['sitelinks']['enwiki']['title'])]
        else:
            return []
    return wikidata_items.flatMap(parse_item_page).collectAsMap()


def extract_items(wikidata_items: RDD, b_property_map: Broadcast, b_item_page_map: Broadcast):
    def parse_item(item):
        property_map = b_property_map.value
        item_page_map = b_item_page_map.value
        if 'enwiki' in item['sitelinks']:
            page_title = item['sitelinks']['enwiki']['title']
        else:
            return None, None

        claims = {}
        for prop_id, property_claims in item['claims'].items():
            if prop_id in property_map:
                prop_name = property_map[prop_id]
                parsed_claims = []
                for c in property_claims:
                    if 'datavalue' in c['mainsnak']:
                        c = c['mainsnak']['datavalue']['value']
                        if type(c) == dict and 'entity-type' in c:
                            claim_item_id = c['id']
                            if claim_item_id in item_page_map:
                                c = item_page_map[c['id']]
                            else:
                                continue
                        parsed_claims.append(c)
                claims[prop_name] = parsed_claims
        return page_title, claims
    return wikidata_items\
        .map(parse_item)\
        .filter(lambda pc: pc[0] is not None)\
        .reduceByKey(lambda x, y: x)\
        .collectAsMap()


def parse_raw_wikidata(output):
    spark_conf = SparkConf().setAppName('QB Wikidata').setMaster(QB_SPARK_MASTER)
    sc = SparkContext.getOrCreate(spark_conf)  # type: SparkContext

    wikidata = sc.textFile('s3a://entilzha-us-west-2/wikidata/wikidata-20170306-all.json')

    def parse_line(line):
        if len(line) == 0:
            return []
        if line[0] == '[' or line[0] == ']':
            return []
        elif line.endswith(','):
            return [json.loads(line[:-1])]
        else:
            return [json.loads(line)]

    parsed_wikidata = wikidata.flatMap(parse_line).cache()
    property_map = extract_property_map(parsed_wikidata)
    b_property_map = sc.broadcast(property_map)

    wikidata_items = parsed_wikidata.filter(lambda d: d['type'] == 'item').cache()
    parsed_wikidata.unpersist()
    item_page_map = extract_item_page_map(wikidata_items)
    b_item_page_map = sc.broadcast(item_page_map)

    parsed_item_map = extract_items(wikidata_items, b_property_map, b_item_page_map)

    with open(output, 'wb') as f:
        pickle.dump({
            'parsed_item_map': parsed_item_map,
            'item_page_map': item_page_map,
            'property_map': property_map
        }, f)

    sc.stop()
