from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from flask import Flask
from flask import request
from extractors.lm import LanguageModel
from util.environment import data_path

app = Flask(__name__)


@app.route('/', methods=['POST'])
def vw_from_title():
    title = request.form['title']
    text = request.form['text']
    return lm.vw_from_title(title, text)

if __name__ == '__main__':
    print('Initializing Language Model')
    lm = LanguageModel(data_path('data/lm.txt'))
    lm.add_corpus('qb')
    lm.add_corpus('wiki')
    lm.add_corpus('source')

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)
    IOLoop.instance().start()
