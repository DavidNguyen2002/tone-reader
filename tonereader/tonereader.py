from nltk.util import ngrams
import spacy
import re
import pandas as pd
from ast import literal_eval

# Cleans comment for parsing


# Remove emojis
def remove_emojis(data: str) -> str:
    emoj = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+",
        re.UNICODE,
    )
    return re.sub(emoj, '', data)


# Remove contractions
def decontracted(phrase: str) -> str:
    # specific phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# Lemmatizes sentence


def lemmatize(text: str, nlp: spacy.Language = spacy.load('en_core_web_sm')) -> list:
    return [token.lemma_ for token in nlp(text)]


def clean_comment(comment: str, nlp: spacy.Language = spacy.load('en_core_web_sm')) -> list:
    # Maybe implement stop word list?

    comment = comment.lower()
    comment = remove_emojis(comment)
    comment = decontracted(comment)
    comment_list = lemmatize(comment, nlp)

    # Remove whitespace - make this better later
    comment_list = [token for token in comment_list if token != ' ']

    return comment_list


def get_ngrams(sentence: list, n: int) -> list:
    final_list = ["<START>" for _ in range(max(1, n - 1))]
    final_list.extend(sentence)
    return list(ngrams(final_list, n))


def csv_to_pd(csv: str) -> pd.DataFrame:
    return pd.read_csv(csv, usecols=['label', 'comment']).dropna()


def read_ngrams(training_data: str, n: int, dest_file: str):
    df_train = csv_to_pd(training_data)
    nlp = spacy.load("en_core_web_sm")

    def add_ngram_row(row, n, nlp) -> list:
        return get_ngrams(clean_comment(row['comment'], nlp), n)

    df_train["ngram"] = df_train.apply(lambda row: add_ngram_row(row, n, nlp), axis=1)
    df_train.to_csv(dest_file, index=False)


def train_ngram(n: int):
    try:
        df_ngram = pd.read_csv(f"tonereader/data/ngrams_{n}.csv", converters={"ngram": literal_eval})
        df_ngram_prev = pd.read_csv(f"tonereader/data/ngrams_{n-1}.csv", converters={"ngram": literal_eval})
    except Exception:
        read_ngrams("train-balanced-sarcasm.csv", n, f"tonereader/data/ngrams_{n}.csv")
        read_ngrams("train-balanced-sarcasm.csv", n - 1, f"tonereader/data/ngrams_{n-1}.csv")
        df_ngram = pd.read_csv(f"tonereader/data/ngrams_{n}.csv", converters={"ngram": literal_eval})
        df_ngram_prev = pd.read_csv(f"tonereader/data/ngrams_{n-1}.csv", converters={"ngram": literal_eval})

    df_ngram_sarcastic = df_ngram.query("label==1")['ngram']
    df_ngram_notsarcastic = df_ngram.query("label==0")['ngram']

    df_ngram_prev_sarcastic = df_ngram_prev.query("label==1")['ngram']
    df_ngram_prev_notsarcastic = df_ngram_prev.query("label==0")['ngram']

    def count_ngram(df: pd.DataFrame):
        return df.explode('ngram').value_counts().rename_axis('unique_values').reset_index(name='counts')

    df_ngram_count_sarcastic = count_ngram(df_ngram_sarcastic)
    df_ngram_count_notsarcastic = count_ngram(df_ngram_notsarcastic)
    df_ngram_prev_count_sarcastic = count_ngram(df_ngram_prev_sarcastic)
    df_ngram_prev_count_notsarcastic = count_ngram(df_ngram_prev_notsarcastic)

    df_ngram_count_sarcastic.to_csv(f"tonereader/data/ngram_sarcastic_count_{n}.csv", index=False)
    df_ngram_count_notsarcastic.to_csv(f"tonereader/data/ngram_notsarcastic_count_{n}.csv", index=False)
    df_ngram_prev_count_sarcastic.to_csv(f"tonereader/data/ngram_prev_sarcastic_count_{n}.csv", index=False)
    df_ngram_prev_count_notsarcastic.to_csv(f"tonereader/data/ngram_prev_notsarcastic_count_{n}.csv", index=False)


# Add smoothing later to take into account OOV words


def is_sarcastic(text: str, n: int = 3):
    df_ngram_count_sarcastic = pd.read_csv(
        f"tonereader/data/ngram_sarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_count_notsarcastic = pd.read_csv(
        f"tonereader/data/ngram_notsarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_prev_count_sarcastic = pd.read_csv(
        f"tonereader/data/ngram_prev_sarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_prev_count_notsarcastic = pd.read_csv(
        f"tonereader/data/ngram_prev_notsarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )

    return is_sarcastic_helper(
        text,
        n,
        df_ngram_count_sarcastic,
        df_ngram_count_notsarcastic,
        df_ngram_prev_count_sarcastic,
        df_ngram_prev_count_notsarcastic,
    )


def is_sarcastic_helper(
    text: str,
    n: int,
    df_ngram_count_sarcastic,
    df_ngram_count_notsarcastic,
    df_ngram_prev_count_sarcastic,
    df_ngram_prev_count_notsarcastic,
    nlp: spacy.Language = spacy.load('en_core_web_sm'),
):
    text_list = clean_comment(text, nlp)
    text_ngrams = get_ngrams(text_list, n)

    # When make this a class, turn it into instance variables

    ngram_sar = df_ngram_count_sarcastic['unique_values'].to_list()
    ngram_notsar = df_ngram_count_notsarcastic['unique_values'].to_list()
    ngram_prev_sar = df_ngram_prev_count_sarcastic['unique_values'].to_list()
    ngram_prev_notsar = df_ngram_prev_count_notsarcastic['unique_values'].to_list()

    def get_index(li: list, val: tuple):
        try:
            return li.index(val)
        except Exception:
            return -1

    prob_sar = 0
    prob_notsar = 0
    for ngram in text_ngrams:
        idx_ngram_sar = get_index(ngram_sar, ngram)
        idx_ngram_notsar = get_index(ngram_notsar, ngram)
        idx_ngram_prev_sar = get_index(ngram_prev_sar, ngram[:-1])
        idx_ngram_prev_notsar = get_index(ngram_prev_notsar, ngram[:-1])

        if (idx_ngram_sar == -1 or idx_ngram_prev_sar == -1) and (
            idx_ngram_notsar == -1 or idx_ngram_prev_notsar == -1
        ):
            continue
        if idx_ngram_notsar == -1 or idx_ngram_prev_notsar == -1:
            sar_numerator = df_ngram_count_sarcastic.loc[idx_ngram_sar, 'counts']
            sar_denominator = df_ngram_prev_count_sarcastic.loc[idx_ngram_prev_sar, 'counts']
            prob_sar += 1.0 * sar_numerator / sar_denominator
        elif idx_ngram_sar == -1 or idx_ngram_prev_sar == -1:
            notsar_numerator = df_ngram_count_notsarcastic.loc[idx_ngram_notsar, 'counts']
            notsar_denominator = df_ngram_prev_count_notsarcastic.loc[idx_ngram_prev_notsar, 'counts']
            prob_notsar += 1.0 * notsar_numerator / notsar_denominator
        else:
            sar_numerator = df_ngram_count_sarcastic.loc[idx_ngram_sar, 'counts']
            sar_denominator = df_ngram_prev_count_sarcastic.loc[idx_ngram_prev_sar, 'counts']
            notsar_numerator = df_ngram_count_notsarcastic.loc[idx_ngram_notsar, 'counts']
            notsar_denominator = df_ngram_prev_count_notsarcastic.loc[idx_ngram_prev_notsar, 'counts']
            prob_sar += 1.0 * sar_numerator / sar_denominator
            prob_notsar += 1.0 * notsar_numerator / notsar_denominator

    if prob_sar == prob_notsar:
        return False
    return True if prob_sar > prob_notsar else False


def ngram_test(test_csv: str, n: int):
    test_file = pd.read_csv(test_csv, usecols=['text', 'sarcastic']).dropna()
    test_file.columns = ['comment', 'label']

    df_ngram_count_sarcastic = pd.read_csv(
        f"tonereader/data/ngram_sarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_count_notsarcastic = pd.read_csv(
        f"tonereader/data/ngram_notsarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_prev_count_sarcastic = pd.read_csv(
        f"tonereader/data/ngram_prev_sarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )
    df_ngram_prev_count_notsarcastic = pd.read_csv(
        f"tonereader/data/ngram_prev_notsarcastic_count_{n}.csv", converters={"unique_values": literal_eval}
    )

    nlp = spacy.load("en_core_web_sm")

    def get_result(row):
        return (
            is_sarcastic_helper(
                row['comment'],
                n,
                df_ngram_count_sarcastic,
                df_ngram_count_notsarcastic,
                df_ngram_prev_count_sarcastic,
                df_ngram_prev_count_notsarcastic,
                nlp,
            )
            == row['label']
        )

    test_file["result"] = test_file.apply(lambda row: get_result(row), axis=1)
    return sum(test_file['result']) / len(test_file.index)
