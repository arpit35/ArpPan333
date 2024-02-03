import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK Punkt tokenizer Models
nltk.download("punkt")

# Download German stopwords
nltk.download("stopwords")
german_stop_words = set(stopwords.words("german"))


def stopwords_remover(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in german_stop_words]
    return " ".join(filtered_text).strip()
