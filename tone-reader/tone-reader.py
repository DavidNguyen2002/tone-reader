from nltk.util import ngrams
import spacy

# Cleans comment for parsing
def clean_comment(comment: str) -> str:
    
    def remove_emojis(s: str) -> str:
        emoji_pattern = re.compile(
            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
            u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
            "+", flags=re.UNICODE
        )
        
        return emoji_pattern.sub(r'', s)
    
    # Remove contractions
    def decontracted(phrase: str) -> list:
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
        nlp = spacy.load('en')
        return [token.lemma_ for token in nlp(str)]
    
    # Maybe implement stop word list?
    
    
    comment = comment.lower()
    comment = remove_emojis(comment)
    comment = decontracted(comment)
    comment_list = lemmatize(comment)
    
    return comment_list
    
def is_sarcastic(text: str) -> bool:
    return '/s' in text
    
def get_ngrams(sentence: list, n: int) -> list:
    final_list = ["<START>" for _ in range(n-1)]
    final_list.extend(list_of_words)
    return list(ngrams(final_list, n))

 