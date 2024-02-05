# Project Overview

This project is a text classification system that uses LightGBM for classification. The project includes scripts for data preprocessing, model training, testing, and a FastAPI application for serving the model. The project also includes Docker support for containerization and GitHub Actions workflows for continuous integration.

## Folder Structure
- [main.py](./main.py) - [🔗](#mainpy)
- [preprocessing](./preprocessing)
  - [main.py](./preprocessing/main.py) - [🔗](#preprocessing-mainpy)
  - [text_sanitization.py](./preprocessing/text_sanitization.py) - [🔗](#text_sanitizationpy)
  - [stopwords_remove.py](./preprocessing/stopwords_remove.py) - [🔗](#stopwords_removepy)
  - [stem_words.py](./preprocessing/stem_words.py) - [🔗](#stem_wordspy)
  - [utils](./preprocessing/utils)
    - [regex.py](./preprocessing/utils/regex.py) - [🔗](#utils-regexpy)
- [model_training](./model_training)
  - [LGBMClassifier.py](./model_training/LGBMClassifier.py) - [🔗](#LGBMClassifier-file)
  - [gridSearch_with_lgbm.py](./model_training/gridSearch_with_lgbm.py) - [🔗](#gridSearch_with_lgbm-file)
- [testing](./testing)
  - [test_text_classifier.py](./testing/test_text_classifier.py) - [🔗](#Testing)
- [app.py](./app.py) - [🔗](#app)
- [Dockerfile](./Dockerfile) - [🔗](#Dockerfile)
- [.github](./.github)
  - [workflows](./.github/workflows)
    - [docker-image.yml](./.github/workflows/docker-image.yml) - [🔗](#docker-imageyml)

# main.py

This is the main script that drives the entire text classification process. It imports necessary modules, reads the data, preprocesses it, trains the model, and saves the trained model and other necessary objects for later use.

## Data Reading and Preprocessing
The script reads the data from a CSV file into a pandas DataFrame. It then preprocesses the data using the `preprocessor` function from the `preprocessing` module. The text data is then converted into numerical data using the `TfidfVectorizer`.

## Label Encoding
The labels are encoded into numerical data using the `LabelEncoder`.

## Train-Test Split
The data is split into training and testing sets using the `train_test_split` function.

## Model Training
The script checks the `INVOKE_GRID_SEARCH` flag. If it's `True`, it invokes the grid search with LightGBM. If it's `False`, it trains the model using the LightGBM classifier.

## Saving the Objects
The trained classifier, the vectorizer, and the label encoder are saved into a file using the `pickle` module. These objects can be loaded later for making predictions.

This script is the main entry point for the text classification process. It orchestrates the entire process from data reading to model training and saving the trained model.


# Preprocessing

The preprocessing package contains several modules that are used to preprocess the text data. These modules include:

## text_sanitization.py

This module contains the `text_sanitizer` function that sanitizes the text data. The function performs the following operations:

- Converts the text to lowercase.
- Removes HTML tags using BeautifulSoup.
- Removes URLs using a regular expression.
- Removes email addresses using a regular expression.
- Replaces the text with an empty string if it contains only numbers.
- Removes special characters and non-German characters.
- Removes extra spaces.
- Returns an empty string if the length of the text is 1.

## stopwords_remove.py

This module contains the `stopwords_remover` function that removes German stopwords from the text. The function tokenizes the text and removes the words that are in the list of German stopwords.

## stem_words.py

This module contains the `stemmer` function that applies stemming to the text. The function uses the SnowballStemmer for German language to stem each word in the text.

## preprocessing-main.py

This module contains the `preprocessor` function that applies all the preprocessing steps to a specified column of a DataFrame. The function applies the `text_sanitizer`, `stopwords_remover`, and `stemmer` functions to the specified column. It also removes the rows with empty strings if the `drop_empty_strings` parameter is `True`.

## utils-regex.py

This module contains several regular expressions that are used in the `text_sanitizer` function. These regular expressions are used to remove URLs, email addresses, special characters, non-German characters, and extra spaces from the text.

The preprocessing package is used to preprocess the text data before it's fed into the machine learning model. The preprocessing steps include text sanitization, stopword removal, and stemming. These steps are crucial for converting the raw text data into a format that can be used by the machine learning model.


# Model Training 

The two Python files you provided are both used for training a LightGBM classifier, but they approach the task in different ways.

## LGBMClassifier file

This file contains the `get_lgbm` function, which trains a LightGBM classifier with a fixed set of hyperparameters. The function takes in the training and testing data, as well as a label encoder (`le`). It then creates an instance of the `LGBMClassifier` with the specified hyperparameters and fits it to the training data. After training, it makes predictions on the test data and prints out the accuracy and the classification report. The trained model is then returned.

## gridSearch_with_lgbm file

This file contains the `get_gridSearch_with_lgbm` function, which uses grid search to find the best hyperparameters for the LightGBM classifier. The function takes in the training and testing data, as well as a label encoder (`le`). It then creates an instance of the `LGBMClassifier` and a grid of hyperparameters to search over. The grid search is performed using 5-fold cross-validation, and the scoring metric is accuracy. The function then fits the grid search object to the training data, which will train a LightGBM classifier for each combination of hyperparameters in the grid and find the best one. After the grid search, it makes predictions on the test data with the best model and prints out the best parameters, the best score, the accuracy, and the classification report. The best model is then returned.

In summary, the `get_lgbm` function trains a LightGBM classifier with a fixed set of hyperparameters, while the `get_gridSearch_with_lgbm` function uses grid search to find the best hyperparameters for the LightGBM classifier.

The `get_gridSearch_with_lgbm` function in the `gridSearch_with_lgbm.py` file uses grid search to find the best hyperparameters for the LightGBM classifier.

Here's a step-by-step breakdown of what the function does:

1. It first sets up a LightGBM model with the objective set to "multiclass" and the number of classes equal to the number of classes in your labels.

2. It then defines a grid of hyperparameters to search over. These include parameters like `max_depth`, `num_leaves`, `class_weight`, `min_split_gain`, `n_estimators`, `learning_rate`, `min_child_samples`, and `reg_alpha`.

3. The function then sets up the grid search with the model and hyperparameter grid. It uses 5-fold cross-validation and aims to maximize accuracy. It also uses all available cores for the computation and outputs progress.

4. The grid search is then fitted to the training data. This will train a LightGBM model for each combination of hyperparameters in the grid and find the best one.

5. After the grid search, the function makes predictions on the test data with the best model and prints out the best parameters, the best score, the accuracy, and the classification report.

6. Finally, the function returns the best model found by the grid search.

# Testing

The `test_text_classifier.py` file contains unit tests for the `/predict_german_text_phrases` endpoint of the REST API. The tests are written using the `unittest` framework. The tests include:

1. `test_predict_german_text_phrases`: This test checks if the API returns a 200 status code when a single text phrase is sent in the request.

2. `test_predict_german_text_phrases_multiple`: This test checks if the API returns a 200 status code when multiple text phrases are sent in the request.

3. `test_predict_german_text_phrases_empty`: This test checks if the API returns a 200 status code when an empty string is sent in the request.

4. `test_predict_german_text_phrases_no_text`: This test checks if the API returns a 400 status code when the request does not contain the "text" key.

5. `test_predict_german_text_phrases_response_format`: This test checks if the API returns a response in the correct format. The response should be a dictionary with the keys "text" and "label", and the values should be lists.

6. `test_predict_german_text_phrases_response_length`: This test checks if the API returns a response with the same length as the input. The lengths of the "text" and "label" lists in the response should be equal to the length of the "text" list in the request.

7. `test_predict_german_text_phrases_response_text`: This test checks if the "text" list in the response is equal to the "text" list in the request.

The tests are run by calling `unittest.main()` in the `__main__` block.

# App 

The `app.py` file is the main entry point for the FastAPI application. It sets up the API and defines the endpoints. Here's a breakdown of the code:

1. `FastAPI Application`: An instance of the FastAPI application is created.

2. `Item Class`: A Pydantic model named `Item` is defined. It has a single attribute `text` which is a list of strings. This model is used for validating the request body.

3. `Loading Saved Objects`: The saved objects are loaded from the `saved_objects.pkl` file. These objects include the classifier, the vectorizer, and the label encoder.

4. `Endpoint Definition`: A POST endpoint `/predict_german_text_phrases` is defined. It takes a dictionary as input, validates the request body using the `Item` model, preprocesses the text, transforms it using the vectorizer, makes a prediction using the classifier, and returns the original text and the predicted labels.

5. `Running the Application`: If the script is run directly, it starts the FastAPI application using `uvicorn.run`. The application runs on `http://127.0.0.1:8000`.

# Docker

## Dockerfile

This file is used to build a Docker image for the application. It starts from an Ubuntu base image, updates the system packages, installs Python3 and pip3, upgrades pip, sets the working directory to `/code`, copies the necessary files and directories into the Docker image, installs the Python dependencies from `requirements.txt`, and finally sets the command to start the FastAPI application using `uvicorn`.

## docker-image.yml

This is a GitHub Actions workflow file that builds and pushes a Docker image to Docker Hub whenever there's a push or pull request to the `main` branch. It checks out the code, sets up Docker Buildx, logs into Docker Hub using secrets, and builds and pushes the Docker image. The Docker image is tagged as `encore7/germanTextClassifier:latest`.