import warnings

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


def get_gridSearch_with_lgbm(x_train, y_train, x_test, y_test, le):

    # Ignore LightGBM warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

    model = LGBMClassifier(objective="multiclass", num_class=len(le.classes_))

    # Define the grid of hyperparameters to search
    hyperparameter_grid = {
        "max_depth": [2, 3, 4, 5, 10, 15],
        "num_leaves": [10, 20, 25, 30, 35, 40, 50, 100, 200],
        "class_weight": ["balanced"],
        "min_split_gain": [0, 10, 20, 40],
        "n_estimators": [2, 5, 20, 40, 30, 50, 100, 150, 200],
        "learning_rate": [0.01, 0.1, 0.15, 0.2],
        "min_child_samples": [10, 20, 30, 40, 50],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    }

    # Set up the grid search
    grid_search = GridSearchCV(
        model,
        param_grid=hyperparameter_grid,
        cv=5,  # Number of cross-validation folds
        scoring="accuracy",  # Can be changed depending on the problem
        n_jobs=-1,  # Use all available cores
        verbose=1,  # Output progress
    )

    # Perform the grid search
    grid_search.fit(x_train, y_train)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return best_model
