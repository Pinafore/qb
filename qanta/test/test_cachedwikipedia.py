import unittest
from qanta.wikipedia.cached_wikipedia import CachedWikipedia

def get_wikidata(self, key: str):
    web_page = "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=" + key + "&language=en&format=json"
    headers = {"Accept": "application/json"}
    req = requests.get(web_page, headers=headers)

    try:
        if not req.json()['search']:
            return None
        else:
            id = req.json()['search'][0]['id']
            url = req.json()['search'][0]['concepturi']
            headers = {"Accept": "application/json"}
            req = requests.get(url, headers=headers)
            claims = req.json()['entities'][id]['claims']
            return claims

    except ValueError:
        return None

class TestCachedWikipedia(unittest.TestCase):
    def test_load_from_memory(self):
        path = "data/external/wikipedia/claims"
        cw = CachedWikipedia(path)
        entity = "emma_watson"
        cw[entity]
        self.assertIn(entity, cw.cache)

    def test_load_from_disk(self):
        path = "data/external/wikipedia/claims"
        cw = CachedWikipedia(path)
        entity = "emma_watson"
        entity_claims_filename = path + "/" + entity + "_claims"
        cw[entity]
        if os.path.exists(entity_claims_filename):
            None
        else:
            self.assertTrue(False)

    def test_load_from_API(self):
        path = "data/external/wikipedia/claims"
        cw = CachedWikipedia(path)
        entity = "albert_einstein"
        entity_claims_filename = path + "/" + entity + "_claims"
        self.assertTrue( cw[entity] is not None)


if __name__ == '__main__':
    unittest.main()
