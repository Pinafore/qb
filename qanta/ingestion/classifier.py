import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from qanta.util.constants import (
    CATEGORIZER_TRAIN_LOCAL_PATH
)

class Classifier:
    def __init__(self, filename=CATEGORIZER_TRAIN_LOCAL_PATH):
        self.vectorizer = TfidfVectorizer(max_df=0.5,
                                          max_features=100000,
                                          stop_words='english',
                                          ngram_range=(1, 2))
        self.category_classifier = MultinomialNB(alpha=0.01)
        self.subcategory_classifier = MultinomialNB(alpha=0.01)
        with open(filename, 'r') as f:
            data = json.load(f)

        texts_with_category = []
        texts_with_subcategory = []
        categories = []
        subcategories = []

        for i in range(len(data["texts"])):
            if data["categories"][i] != "None":
                texts_with_category.append(data["texts"][i])
                categories.append(data["categories"][i])
            if data["subcategories"][i] != "None":
                texts_with_subcategory.append(data["texts"][i])
                subcategories.append(data["subcategories"][i])

        self.vectorizer.fit(texts_with_category)

        texts_with_category = self.vectorizer.transform(texts_with_category)
        texts_with_subcategory = self.vectorizer.transform(texts_with_subcategory)
        self.category_classifier.fit(texts_with_category, categories)
        self.subcategory_classifier.fit(texts_with_subcategory, subcategories)

    def predict_category(self, question):
        category = self.predict_categories([question])
        return category[0] if category else None

    def predict_subcategory(self, question):
        subcategory = self.predict_subcategories([question])
        return subcategory[0] if subcategory else None

    def predict_categories(self, questions):
        X = self.vectorizer.transform(questions)
        return self.category_classifier.predict(X)

    def predict_subcategories(self, questions):
        X = self.vectorizer.transform(questions)
        return self.subcategory_classifier.predict(X)
