from typing import List, NamedTuple, Tuple
from py4j.java_gateway import JavaGateway


Mention = NamedTuple('Mention', [
    ('page_id', int),
    ('page', str),
    ('start', int),
    ('end', int),
    ('rho', float)
])



class TagmeClient:
    def __init__(self):
        self.gateway = JavaGateway()

    def tag_mentions(self, text_list: List[str], threshold=.15) -> Tuple[List[str], List[List[Mention]]]:
        j_list = self.gateway.jvm.java.util.ArrayList()
        for t in text_list:
            j_list.append(t)

        results = self.gateway.entry_point.annotateMany(j_list)
        clean_text = []
        all_mentions = []
        for r in results:
            clean_text.append(r.getText())
            annotations = r.getAnnotations()
            mentions = [Mention(
                a.getTopic(),
                self.id_to_page(a.getTopic()),
                a.getStart(),
                a.getEnd(),
                a.getRho()
            ) for a in annotations if a.getRho() > threshold]
            all_mentions.append(mentions)

        return clean_text, all_mentions

    def id_to_page(self, topic):
        page = self.gateway.entry_point.idToPage(topic)
        if page is None:
            return ''
        else:
            return page.replace(' ', '_')

