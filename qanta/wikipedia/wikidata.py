import json
import pickle
from abc import ABCMeta, abstractmethod
from collections import namedtuple

from pyspark import SparkConf, SparkContext, RDD, Broadcast
from qanta.util.environment import QB_SPARK_MASTER


TimeData = namedtuple('TimeData', 'after before calendarmodel precision time timezone')
QuantityData = namedtuple('QuantityData', 'amount unit upperbound lowerbound')
GlobeCoordinateData = namedtuple('GlobeCoordinateData', 'globe latitude longitude altitude precision')


class WikiDatatype(metaclass=ABCMeta):
    @property
    @abstractmethod
    def datatype(self):
        pass

    @staticmethod
    @abstractmethod
    def parse(datavalue):
        pass


class WikiString(WikiDatatype):
    @property
    def datatype(self):
        return 'string'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']


class WikiTime(WikiDatatype):
    @property
    def datatype(self):
        return 'time'

    @staticmethod
    def parse(datavalue):
        value = datavalue['value']
        return TimeData(
            value['after'], value['before'],
            value['calendarmodel'], value['precision'],
            value['time'], value['timezone']
        )


class WikiItem(WikiDatatype):
    @property
    def datatype(self):
        return 'wikibase-item'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']['id']


class WikiProperty(WikiDatatype):
    @property
    def datatype(self):
        return 'wikibase-property'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']['id']


class WikiExternalId(WikiDatatype):
    @property
    def datatype(self):
        return 'external-id'

    @staticmethod
    def parse(datavalue):
        """
        We don't have a way to make use of it so just don't parse it
        :param datavalue: 
        :return: 
        """
        return None


class WikiMonolingualText(WikiDatatype):
    @property
    def datatype(self):
        return 'monolingualtext'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']['text']


class WikiCommonsMedia(WikiDatatype):
    @property
    def datatype(self):
        return 'commonsMedia'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']


class WikiQuantity(WikiDatatype):
    @property
    def datatype(self):
        return 'quantity'

    @staticmethod
    def parse(datavalue):
        value = datavalue['value']
        return QuantityData(value.get('amount'), value.get('unit'), value.get('upperbound'), value.get('lowerbound'))


class WikiGlobeCoordinate(WikiDatatype):
    @property
    def datatype(self):
        return 'globe-coordinate'

    @staticmethod
    def parse(datavalue):
        value = datavalue['value']
        return GlobeCoordinateData(
            value.get('globe'), value.get('latitude'), value.get('longitude'),
            value.get('altitude'), value.get('precision')
        )


class WikiUrl(WikiDatatype):
    @property
    def datatype(self):
        return 'url'

    @staticmethod
    def parse(datavalue):
        return datavalue['value']


class WikiMath(WikiDatatype):
    @property
    def datatype(self):
        return 'math'

    @staticmethod
    def parse(datavalue):
        """
        We don't have a way to use this so parse it into nothing. To parse this out is just datavalue['value']
        :param datavalue: 
        :return: 
        """
        return None

wiki_datatypes = [
    WikiString(), WikiTime(),
    WikiItem(), WikiProperty(), WikiExternalId(),
    WikiMath(), WikiUrl(),
    WikiMonolingualText(), WikiCommonsMedia(), WikiQuantity(), WikiGlobeCoordinate()
]


datatype_parsers = {}
for wd in wiki_datatypes:
    datatype = wd.datatype
    datatype_parsers[datatype] = wd.parse


class Claim:
    def __init__(self, subject, relation, wiki_object, datatype, title, property_id):
        self.subject = subject
        self.relation = relation
        self.object = wiki_object
        self.datatype = datatype
        self.title = title
        self.property_id = property_id

    def __repr__(self):
        return '(s="{}", r="{}", o="{}", dt="{}", title="{}"'.format(
            self.subject, self.relation, self.object, self.datatype, self.title
        )


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


def extract_claims(wikidata_items: RDD, b_property_map: Broadcast):
    def parse_item_claims(item):
        item_id = item['id']
        property_map = b_property_map.value
        if 'enwiki' in item['sitelinks']:
            title = item['sitelinks']['enwiki']['title']
        else:
            title = None
        item_claims = []
        for property_id, property_claims in item['claims'].items():
            if property_id in property_map:
                property_name = property_map[property_id]
                for claim in property_claims:
                    mainsnak = claim['mainsnak']
                    if 'datatype' in mainsnak and 'datavalue' in mainsnak:
                        datatype = mainsnak['datatype']
                        datavalue = mainsnak['datavalue']
                        if datatype in datatype_parsers:
                            wiki_object = datatype_parsers[datatype](datavalue)
                            if wiki_object is not None:
                                item_claims.append(
                                    Claim(item_id, property_name, wiki_object, datatype, title, property_id)
                                )

        return item_claims

    return wikidata_items.flatMap(parse_item_claims)


def extract_claim_types(wikidata_items: RDD):
    def parse_types(item):
        value_types = []
        for property_claims in item['claims'].values():
            for c in property_claims:
                mainsnak = c['mainsnak']
                if 'datatype' in mainsnak:
                    value_types.append(mainsnak['datatype'])
        return value_types

    return set(wikidata_items.flatMap(parse_types).distinct().collect())


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
