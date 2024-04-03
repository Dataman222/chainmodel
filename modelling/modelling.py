from model.SGD import SGD
from model.randomforest import RandomForest
from model.adaboost import AdaBoost
from model.voting import Voting
from model.hist_gb import Hist_GB
from model.random_trees_ensembling import RandomTreesEmbedding

def model_predict(data, df, name):
    results = []

    # Start of new code for chained classification
    print("Beginning chained classification...")
    y_train_chained = data.get_y_train_chained()  # Accessing chained training labels
    y_test_chained = data.get_y_test_chained()  # Accessing chained test labels

    # Mapping label sets to the appropriate columns in y_train_chained and y_test_chained
    label_sets = {
        'Type_2': 0,  # Index for Type 2 labels
        'Type_2_3': 1,  # Index for Type 2+3 labels
        'Type_2_3_4': 2,  # Index for Type 2+3+4 labels
    }

    for label_type, index in label_sets.items():
        print(f"Evaluating {label_type}")
        y_train = y_train_chained[:, index]  # Selecting the specific set of labels for training
        y_test = y_test_chained[:, index]  # Selecting the specific set of labels for testing

        # Using RandomForest as an example for each type of classification
        model = RandomForest(f"RandomForest_{label_type}", data.get_embeddings(), y_train)
        model.train(data.X_train, y_train)
        y_pred = model.predict(data.X_test)
        model.print_results(y_test, y_pred)
        results.append((label_type, model.evaluate(y_test, y_pred)))
    # End of new code for chained classification

###

def model_evaluate(model, data):
    model.print_results(data)
