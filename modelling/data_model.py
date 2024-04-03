import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)

class MultilabelData():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        ### Initial processing
        X_DL = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X_DL = X_DL.to_numpy()
        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X_DL[y_series.isin(good_y_value)]  # Adjusted to use X_DL

        new_test_size = X_DL.shape[0] * 0.2 / X_good.shape[0]  # Adjusted to use X_DL

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good
        )
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X_good  # Adjusted to use X_good

        ### New multilabel code
        self.X = X_DL  # Adjusted to use X_DL
        self.df = df
        self.df['Type_2'] = self.df[Config.TYPE_COLS[0]]
        self.df['Type_2_3'] = self.df.apply(lambda x: '+'.join(filter(pd.notnull, [x[Config.TYPE_COLS[0]], x[Config.TYPE_COLS[1]]])), axis=1)
        self.df['Type_2_3_4'] = self.df.apply(lambda x: '+'.join(filter(pd.notnull, [x[Config.TYPE_COLS[0]], x[Config.TYPE_COLS[1]], x[Config.TYPE_COLS[2]]])), axis=1)

        self.y_chained = self.df[['Type_2', 'Type_2_3', 'Type_2_3_4']].to_numpy()
        self.X_train_chained, self.X_test_chained, self.y_train_chained, self.y_test_chained = train_test_split(
            self.X, self.y_chained, test_size=0.2, random_state=0
        )

        # Analyzing label distribution after initialization
        self.analyze_label_distribution()

    def get_analyze_label_distribution(self): # review arguments
        # the return below needs to be reviewed
        return self.y_train, self.y_test, self.y_train_chained, self.y_test_chained
    # end new multilabel code

    ### new multilabel methods
    def get_y_train_chained(self):
        return self.y_train_chained

    ### new multilabel
    def get_y_test_chained(self):
        return self.y_test_chained

    ### from here as in multiclass original methods
    def get_embeddings(self):
        return self.X

    def get_embeddings(self):
        return  self.embeddings

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_train_df(self):
        return self.train_df

    def get_type_test_df(self):
        return self.test_df

    def get_X_DL_test(self):
        return self.X_DL_test

    def get_X_DL_train(self):
        return self.X_DL_train
