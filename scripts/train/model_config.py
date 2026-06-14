# scripts/train/model_config.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

MODEL_REGISTRY = {
    "0": {
        "algo_name": "RandomForest",
        "class": RandomForestClassifier,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 2,
            "random_state": 42,
            "n_jobs": -1
        }
    },
    "1": {
        "algo_name": "GradientBoosting",
        "class": GradientBoostingClassifier,
        "hyperparameters": {
            "n_estimators": 80,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42
        }
    },
    "2": {
        "algo_name": "LogisticRegression",
        "class": LogisticRegression,
        "hyperparameters": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42
        }
    }
}