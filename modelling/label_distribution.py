import numpy as np
import pandas as pd
from Config import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def analyze_labels(self.y_train, self.y_test, self.y_train_chained, self.y_test_chained): # PG: review arguments
    print("Analyzing label distribution...")
    if self.y_train is not None:
        print("Original Label Distribution in Training Set:")
        multiclass_label_train = np.sum(self.y_train, axis=0)
        print(multiclass_label_train)

    if self.y_test is not None:
        print("Original Label Distribution in Test Set:")
        multiclass_label_test = np.sum(self.y_test, axis=0)
        print(multiclass_label_test)

    if self.y_train_chained is not None:
        print("Chained Label Distribution in Training Set:")
        chained_label_train = np.sum(self.y_train_chained, axis=0)
        print(chained_label_train)

    if self.y_test_chained is not None:
        print("Chained Label Distribution in Test Set:")
        chained_label_test = np.sum(self.y_test_chained, axis=0)
        print(chained_label_test)

    return multiclass_label_train, multiclass_label_test, chained_label_train, chained_label_test
