import unittest
from unittest import TestCase

import requests

PORT = 8000
URL = "http://127.0.0.1:" + str(PORT)
APP_URL = URL + "/predict_german_text_phrases"


class TryTesting(TestCase):
    # call the api and check if the response is 200
    def test_predict_german_text_phrases(self):
        item = {"text": ["testing"]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)

    # test with multiple phrases
    def test_predict_german_text_phrases_multiple(self):
        item = {"text": ["testing1", "testing2", "testing3"]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)

    # test with empty string
    def test_predict_german_text_phrases_empty(self):
        item = {"text": [""]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)

    # test with no text key in json
    # expecting a bad request error
    def test_predict_german_text_phrases_no_text(self):
        item = {}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 400)

    # test with invalid response json
    def test_predict_german_text_phrases_response_format(self):
        item = {"text": ["testing"]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIsInstance(response_json, dict)
        self.assertListEqual(list(response_json.keys()), ["text", "label"])
        self.assertIsInstance(response_json["text"], list)
        self.assertIsInstance(response_json["label"], list)

    # test if the lenght of the response is equal to the length of the input
    def test_predict_german_text_phrases_response_length(self):
        item = {"text": ["testing1", "testing2", "testing3"]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual(len(item["text"]), len(response_json["text"]))
        self.assertEqual(len(item["text"]), len(response_json["label"]))

    # test if the response text is equal to the input text
    def test_predict_german_text_phrases_response_text(self):
        item = {"text": ["testing1", "testing2", "testing3"]}
        response = requests.post(APP_URL, json=item)
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertListEqual(item["text"], response_json["text"])


if __name__ == "__main__":
    unittest.main()
