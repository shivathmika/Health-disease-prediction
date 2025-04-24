# param_distributions.py

param_distributions = {
    "decision_tree_gini": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "decision_tree_entropy": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "random_forest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "extra_trees": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5]
    },
    "ada_boost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1]
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["saga"]
    },
    "svm_linear": {
        "C": [0.1, 1, 10]
    },
    "svm_rbf": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"]
    },
    "naive_bayes": {},  # No major hyperparameters for GaussianNB
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    "lightgbm": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    },
    "catboost": {
        "iterations": [100, 200],
        "depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2]
    }
}
