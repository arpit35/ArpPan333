import re

from bs4 import BeautifulSoup

import preprocessing.utils.regex as reg


def text_sanitizer(text):
    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove URLs
    text = re.sub(reg.REMOVE_URLS, "", text)

    # Remove email addresses
    text = re.sub(reg.REMOVE_EMAIL_ADDRESSES, "", text)

    # Replace with empty string if it contains only numbers
    text = re.sub(reg.REPLACE_WITH_EMPTY_STRING_IF_ONLY_CONTAINS_NUMBERS, "", text)

    # Remove special characters and non-german characters
    text = re.sub(reg.REMOVE_SPECIAL_CHAR_AND_NON_GERMAN_CHAR, "", text)

    # Remove extra spaces
    text = re.sub(reg.REMOVE_EXTRA_SPACES, " ", text).strip()

    # Return empty string if the length is 1
    if len(text) == 1:
        return ""

    return text
