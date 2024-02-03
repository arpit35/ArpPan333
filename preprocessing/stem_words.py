from nltk.stem.snowball import SnowballStemmer

stem = SnowballStemmer("german")

print(stem)


def stemmer(text):
    return " ".join([stem.stem(word) for word in text.split()]).strip()
