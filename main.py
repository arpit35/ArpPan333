import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import model.gridSearch_with_lgbm as gridSearch_with_lgbm
import model.LGBMClassifier as LGBMClassifier
import preprocessing.main as preprocessing

INVOKE_GRID_SEARCH = False

# Read the CSV file into a DataFrame
df = pd.read_csv(r"sample_data.csv")

df = preprocessing.preprocessor(df, "text", True)
# Convert text data into numerical data
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["text"])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

if INVOKE_GRID_SEARCH:
    get_gridSearch_with_lgbm = gridSearch_with_lgbm.get_gridSearch_with_lgbm(
        x_train, y_train, x_test, y_test, le
    )
else:
    classifier = LGBMClassifier.get_lgbm(x_train, y_train, x_test, y_test, le)

    objects_to_save = {
        "classifier": classifier,
        "vectorizer": vectorizer,
        "label_encoder": le,
    }

    # Save all objects to a file
    with open("saved_objects.pkl", "wb") as f:
        pickle.dump(objects_to_save, f)
