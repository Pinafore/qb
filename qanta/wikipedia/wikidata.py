import json
import pickle
from abc import ABCMeta, abstractmethod
from collections import namedtuple, Counter, defaultdict

from pyspark import SparkConf, SparkContext, RDD, Broadcast
from qanta.util.environment import QB_SPARK_MASTER
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.wikipedia.cached_wikipedia import normalize_wikipedia_title


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

Claim = namedtuple('Claim', 'item property object datatype title property_id item_id')


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


def extract_item_map(wikidata_items: RDD):
    def parse_item(item):
        if 'en' in item['labels']:
            label = item['labels']['en']['value']
            return item['id'], label
        else:
            return None
    return wikidata_items.map(parse_item).filter(lambda i: i is not None).collectAsMap()


def extract_claims(wikidata_items: RDD, b_property_map: Broadcast, b_item_map: Broadcast):
    def parse_item_claims(item):
        item_id = item['id']
        item_map = b_item_map.value
        if item_id not in item_map:
            return []
        item_label = item_map[item_id]
        property_map = b_property_map.value
        if 'enwiki' in item['sitelinks']:
            title = item['sitelinks']['enwiki']['title']
        else:
            title = None
        item_claims = []
        for property_id, property_claims in item['claims'].items():
            if property_id in property_map:
                if property_id not in property_map:
                    continue
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
                                    Claim(item_label, property_name, wiki_object, datatype, title, property_id, item_id)
                                )
        return item_claims

    return wikidata_items.flatMap(parse_item_claims)


def clean_claims(claims: RDD, b_item_map: Broadcast):
    def clean(claim):
        item_map = b_item_map.value
        if claim.datatype == 'wikibase-item':
            if claim.object in item_map:
                claim = claim._replace(object=item_map[claim.object])
                return claim
            else:
                return None
        elif claim.datatype == 'quantity':
            unit = claim.object.unit
            unit = unit.split('/')[-1]
            if unit in item_map:
                claim = claim._replace(object=item_map[unit])
                return claim
            else:
                return None
        return claim

    dt_filter = {'wikibase-item', 'string', 'monolingualtext', 'quantity', 'time'}

    return claims.filter(lambda c: c.datatype in dt_filter).map(clean).filter(lambda c: c is not None)


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

object_blacklist = {
    'Wikimedia disambiguation page',
    'common name',
    'male given name',
    'Wikimedia list article'
}

object_merge_map = {
    'sovereign state': 'country',
    'member state of the United Nations': 'country',
    'member state of the European Union': 'country',
    'member state of the Council of Europe': 'country',
    'permanent member of the United Nations Security Council': 'country',
    'landlocked country': 'country',
    'island nation': 'country',
    'republic': 'country',
    'former country': 'country',
    'unitary state': 'country',
    'communist state': 'country',
    'Commonwealth realm': 'country',
    'secular state': 'country',
    'social state': 'country',
    'legal state': 'country',
    'dominion': 'country',
    'member state of Mercosur': 'country',
    'Islamic republic': 'country',
    'member state of the Union of South American Nations': 'country',
    'federal republic': 'country',
    'constitutional republic': 'country',
    'state': 'country',
    'city with millions of inhabitants': 'city',
    'big city': 'city',
    'capital': 'city',
    'port city': 'city',
    'state or insular area capital in the United States': 'city',
    'metropolis': 'city',
    'one-act play': 'play',
    'inner planet': 'planet',
    'outer planet': 'planet',
    'vector physical quantity': 'physical quantity',
    'scalar physical quantity': 'physical quantity',
    'extensive physical property': 'physical quantity',
    'intensive property': 'physical quantity',
    'thermodynamic system property (state function)': 'physical quantity',
    'conflict': 'war'
}


def is_god(obj):
    if obj == 'mythological Greek character' or obj == 'Twelve Olympians':
        return True
    elif 'deity' in obj or 'deities' in obj or 'god' in obj:
        return True
    else:
        return False


def create_instance_of_map(wikidata_claims_instance_of_path, output_path, n_types=50):
    ds = QuizBowlDataset(1, guesser_train=True)
    training_data = ds.training_data()
    answers = set(training_data[1])
    answer_counts = Counter(training_data[1])
    with open(wikidata_claims_instance_of_path) as f:
        claims = defaultdict(set)
        for line in f:
            c = json.loads(line)
            if c['title'] is not None:
                title = normalize_wikipedia_title(c['title'])
                if title in answers:
                    c_object = c['object']
                    if c_object in object_blacklist:
                        continue
                    if c_object in object_merge_map:
                        claims[title].add(object_merge_map[c_object])
                    elif is_god(c_object):
                        claims[title].add('god')
                    else:
                        claims[title].add(c_object)
    for k in claims:
        if 'literary work' in claims[k] and len(claims[k]) > 1:
            claims[k].remove('literary work')

    total_counts = defaultdict(int)
    class_counts = defaultdict(int)
    answer_types = {}
    for a in answers:
        if a in claims:
            answer_types[a] = claims[a]
            for obj in claims[a]:
                total_counts[obj] += answer_counts[a]
                class_counts[obj] += 1

    sorted_total_counts = sorted(total_counts.items(), reverse=True, key=lambda k: k[1])
    top_types = {key for key, count in sorted_total_counts[:n_types]}
    instance_of_map = {}
    for a in answer_types:
        top_class_types = top_types.intersection(answer_types[a])
        if len(top_class_types) == 0:
            instance_of_map[a] = 'NO MATCH!'
        elif len(top_class_types) > 1:
            frequency = 0
            final_a_type = None
            for a_type in top_class_types:
                if class_counts[a_type] > frequency:
                    frequency = class_counts[a_type]
                    final_a_type = a_type
            instance_of_map[a] = final_a_type
        else:
            instance_of_map[a] = next(iter(top_class_types))

    with open(output_path, 'wb') as f:
        pickle.dump(instance_of_map, f)
