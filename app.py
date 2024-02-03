import json
import pickle
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import preprocessing.main as preprocessing

app = FastAPI()


class Item(BaseModel):
    text: List[str]


# Load all objects
with open("saved_objects.pkl", "rb") as f:
    loaded_objects = pickle.load(f)

# Access the objects
classifier = loaded_objects["classifier"]
vectorizer = loaded_objects["vectorizer"]
le = loaded_objects["label_encoder"]


@app.post("/predict_german_text_phrases")
def predict_german_text_phrases(item: Item):
    try:
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(item.text, columns=["text"])
        df_preprocessed = preprocessing.preprocessor(df.copy(), "text", False)
        x = vectorizer.transform(df_preprocessed["text"])
        prediction = classifier.predict(x)

        return {
            "text": df["text"].tolist(),
            "label": le.inverse_transform(prediction).tolist(),
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")


# Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
