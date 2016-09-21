from sklearn.base import TransformerMixin


class DFColumnTransformer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None, **fit_params):
        if len(self.columns) == 1:
            return X[self.columns[0]]
        else:
            return X[self.columns]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {'columns': self.columns}


class CSCTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.tocsc()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}