import preprocessing.stem_words as stem_words
import preprocessing.stopwords_remove as stopwords_remove
import preprocessing.text_sanitization as text_sanitization


def preprocessor(df, column_name, drop_empty_strings=True):
    df[column_name] = (
        df[column_name]
        .apply(text_sanitization.text_sanitizer)
        .apply(stopwords_remove.stopwords_remover)
        .apply(stem_words.stemmer)
    )

    # Remove rows with empty strings
    if drop_empty_strings:
        return df[df[column_name].ne("")]
    return df
