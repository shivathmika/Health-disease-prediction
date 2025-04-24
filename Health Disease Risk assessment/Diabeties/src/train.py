import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
import pickle

from config import TRAINING_FILE_WITH_FOLDS, MODEL_OUTPUT
import model_dispacher, param_distributions

from model_dispacher import models
from param_distributions import param_distributions

# Load the dataset with folds
data = pd.read_csv(TRAINING_FILE_WITH_FOLDS)

# Tune model using RandomizedSearchCV
def tune_model(model_name, X, y):
    model = models[model_name]
    if model_name in param_distributions:
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[model_name],
            n_iter=10,
            cv=5,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    else:
        model.fit(X, y)
        return model, None

# Evaluate model on each fold
def evaluate_model_on_folds(model, model_name):
    f1_scores = []
    for fold in range(5):
        df_train = data[data.kfold != fold].reset_index(drop=True)
        df_valid = data[data.kfold == fold].reset_index(drop=True)

        X_train = df_train.drop(["Outcome", "kfold"], axis=1).values
        y_train = df_train.Outcome.values
        X_valid = df_valid.drop(["Outcome", "kfold"], axis=1).values
        y_valid = df_valid.Outcome.values

        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        f1 = f1_score(y_valid, preds)
        print(f"📁 Fold {fold}: F1 Score = {f1:.4f}")
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"🎯 Avg F1 Score for {model_name}: {avg_f1:.4f}")
    return avg_f1, f1_scores

if __name__ == "__main__":
    X = data.drop("Outcome", axis=1).values
    y = data.Outcome.values

    best_overall_model = None
    best_overall_score = 0
    best_overall_model_name = ""
    best_overall_params = None

    for model_name in models.keys():
        print(f"\n🔍 Tuning model: {model_name}")
        best_model, best_params = tune_model(model_name, X, y)
        print(f"✅ Best Params: {best_params if best_params else 'Default'}")

        print(f"📊 Evaluating {model_name} on each fold...")
        avg_f1, _ = evaluate_model_on_folds(best_model, model_name)

        if avg_f1 > best_overall_score:
            best_overall_score = avg_f1
            best_overall_model = best_model
            best_overall_model_name = model_name
            best_overall_params = best_params

    # Save the best overall model
    final_model_path = f"{MODEL_OUTPUT}/{best_overall_model_name}_BEST.pkl"
    with open(final_model_path, "wb") as f:
        pickle.dump(best_overall_model, f)

    print(f"\n🏆 Best Overall Model: {best_overall_model_name}")
    print(f"💯 Best F1 Score: {best_overall_score:.4f}")
    print(f"🧪 Best Params: {best_overall_params if best_overall_params else 'Default'}")
    print(f"💾 Model saved to {final_model_path}")