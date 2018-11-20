import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from qanta.datasets.protobowl import load_protobowl

import matplotlib
matplotlib.use('agg')
from plotnine import ggplot, aes, geom_point, stat_function, labs

report_dir = 'output/reporting'
if not os.path.isdir(report_dir):
    os.mkdir(report_dir)


class CurveScore:

    def __init__(self):
        ckp_dir = os.path.join(report_dir, 'curve_pipeline.pkl')
        if os.path.isfile(ckp_dir):
            print('loading pipeline')
            with open(ckp_dir, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            print('fitting pipeline')
            self.pipeline = self.fit_curve()
            with open(ckp_dir, 'wb') as f:
                pickle.dump(self.pipeline, f)

    def get_weight(self, x):
        return self.pipeline.predict(np.asarray([[x]]))[0]

    def score(self, guesses, question):
        '''guesses is a list of {'guess': GUESS, 'buzz': True/False}
        '''
        char_length = len(question['text'])
        buzzes = [x['buzz'] for x in guesses]
        if True not in buzzes:
            return 0
        buzz_index = buzzes.index(True)
        rel_position = (1.0 * guesses[buzz_index]['char_index']) / char_length
        weight = self.get_weight(rel_position)
        result = guesses[buzz_index]['guess'] == question['page']
        return weight * result

    def score_optimal(self, guesses, question):
        '''score with an optimal buzzer'''
        char_length = len(question['text'])
        buzz_index = char_length
        for g in guesses:
            if g['guess'] == question['page']:
                buzz_index = g['char_index']
                break
        rel_position = (1.0 * buzz_index) / char_length
        return self.get_weight(rel_position)

    def score_stable(self, guesses, question):
        '''score with an optimal buzzer'''
        char_length = len(question['text'])
        buzz_index = char_length
        for g in guesses[::-1]:
            if g['guess'] != question['page']:
                buzz_index = g['char_index']
                break
        rel_position = (1.0 * buzz_index) / char_length
        return self.get_weight(rel_position)

    def fit_curve(self):
        df, questions = load_protobowl()
        # convert prompt to false
        df.result = df.result.apply(lambda x: x is True)

        xy = list(zip(df.relative_position.tolist(), df.result.tolist()))
        xy = sorted(xy, key=lambda x: x[0])
        ratios = dict()
        cnt = 0
        for x, y in xy:
            x = int(x*1000)
            ratios[x] = cnt
            cnt += y
        ratios = sorted(ratios.items(), key=lambda x: x[0])
        ratios = [(x / 1000, y) for x, y in ratios]

        ttl_correct = df.result.tolist().count(True)
        ttl_correct = len(xy)
        curve = [(x, 1 - y / ttl_correct) for x, y in ratios]
        X, y = list(map(list, zip(*curve)))

        X = np.asarray(X)
        y = np.asarray(y)
        degree = 3
        polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)
        print(pipeline.steps[1][1].coef_)

        def get_weight(x):
            return pipeline.predict(np.asarray([[x]]))[0]

        ddf = pd.DataFrame({'x': X, 'y': y})
        p0 = ggplot(ddf, aes(x='x', y='y')) \
            + geom_point(size=0.3, color='blue', alpha=0.5, shape='+') \
            + stat_function(fun=get_weight, color='red', size=2, alpha=0.5) \
            + labs(x='Position', y='Weight')
        p0.save('output/reporting/curve_score.pdf')
        p0.draw()

        return pipeline


if __name__ == '__main__':
    curve_score = CurveScore()
