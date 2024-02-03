import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report


def get_lgbm(x_train, y_train, x_test, y_test, le):
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        max_depth=7,
        num_leaves=100,
        class_weight="balanced",
        min_split_gain=40,
        n_estimators=150,
        learning_rate=0.1,
    )
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model
