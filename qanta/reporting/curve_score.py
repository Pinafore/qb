import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
from plotnine import ggplot, aes, geom_point, stat_function, labs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from qanta.datasets.protobowl import load_protobowl
from qanta.buzzer.util import output_dir


class CurveScore:

    def __init__(self):
        ckp_dir = os.path.join(output_dir, 'curve_pipeline.pkl')
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
