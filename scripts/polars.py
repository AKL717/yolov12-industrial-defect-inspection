import pandas as pd


class _Frame:
    def __init__(self, dataframe):
        self._dataframe = dataframe

    def to_dict(self, as_series=False):
        if as_series:
            return self._dataframe.to_dict()
        return {column: self._dataframe[column].tolist() for column in self._dataframe.columns}


def read_csv(path, infer_schema_length=None):
    return _Frame(pd.read_csv(path))
