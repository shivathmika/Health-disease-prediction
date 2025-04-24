# model_dispatcher.py
from sklearn import tree, ensemble, neighbors, linear_model, svm, naive_bayes
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

models = {
    # Decision Trees
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
    
    # Ensemble Models
    "random_forest": ensemble.RandomForestClassifier(),
    "gradient_boosting": ensemble.GradientBoostingClassifier(),
    "extra_trees": ensemble.ExtraTreesClassifier(),
    "ada_boost": ensemble.AdaBoostClassifier(),
    
    # K-Nearest Neighbors
    "knn": neighbors.KNeighborsClassifier(),
    
    # Linear Models
    "logistic_regression": linear_model.LogisticRegression(),
    
    # Support Vector Machines
    "svm_linear": svm.SVC(kernel="linear"),
    "svm_rbf": svm.SVC(kernel="rbf"),
    
    # Naive Bayes
    "naive_bayes": naive_bayes.GaussianNB(),
    
    # Gradient Boosting Libraries
    "xgboost": xgb.XGBClassifier(eval_metric="logloss"),
    "lightgbm": lgb.LGBMClassifier(),
    "catboost": cb.CatBoostClassifier()
}
