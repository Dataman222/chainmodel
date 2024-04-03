from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
from modelling.label_distribution import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    #load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def analyze_label_distribution():
    multiclass_label_train, multiclass_label_test, chained_label_train, chained_label_test= analyze_labels()
    return multiclass_label_train, multiclass_label_test, chained_label_train, chained_label_test # review

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    grouped_df = df.groupby(Config.GROUPED) # grouping will need to be reviewed to consider all labels, not just y1
    labels = analyze_label_distribution()  # review, not sure whether this method call goes here in the main block or the variable declaration is correct.

    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        data = get_data_object(X, group_df)
        perform_modelling(data, group_df, name)
