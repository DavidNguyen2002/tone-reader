from nltk.util import ngrams
import spacy
import re

# Cleans comment for parsing
# This is so ugly. clean immediately

# Remove emojis
def remove_emojis(data: str) -> str:
    emoj = re.compile("["
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
                      "]+", re.UNICODE)
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

def lemmatize(text: str) -> list:
    nlp = spacy.load("en_core_web_sm")
    return [token.lemma_ for token in nlp(text)]
    
def clean_comment(comment: str) -> str:
    # Maybe implement stop word list?
    
    comment = comment.lower()
    comment = remove_emojis(comment)
    comment = decontracted(comment)
    comment_list = lemmatize(comment)
    
    # Remove whitespace - make this better later
    comment_list = [token for token in comment_list if token != ' ']
    
    return comment_list
    
def is_sarcastic(text: str) -> bool:
    return '/s' in text
    
def get_ngrams(sentence: list, n: int) -> list:
    final_list = ["<START>" for _ in range(n-1)]
    final_list.extend(sentence)
    return list(ngrams(final_list, n))