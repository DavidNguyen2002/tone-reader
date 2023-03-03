from tonereader.tonereader import *
from unittest.mock import patch

def test_remove_emoji():
    test = "Examples of emoji are 😂, 😃, 🧘🏻‍♂️, 🌍, 🌦️, 🥖, 🚗, 📱, 🎉, ❤️, ✅, and 🏁."
    assert remove_emojis(test) == "Examples of emoji are , , , , , , , , , , , and ."
    
def test_decontracted():
    test = "What's up? My name's David. What're you up to?"
    assert decontracted(test) == "What is up? My name is David. What are you up to?"
    
def test_lemmatize():
    test = "You were going to the store. I ate an apple."
    assert lemmatize(test) == ["you", "be", "go", "to", "the", "store", ".", "I", "eat", "an", "apple", "."]
    
def test_is_sarcastic():
    test1 = "Yeah, I love spending time with you. /s"
    test2 = "I love open source"
    assert is_sarcastic(test1)
    assert not is_sarcastic(test2)
    
def test_get_ngrams():
    test = ["Test", "sentence", "is", "very", "cool"]
    assert get_ngrams(test, 3) == [("<START>", "<START>", "Test"), ("<START>", "Test", "sentence"), ("Test", "sentence", "is"), ("sentence", "is", "very"), ("is", "very", "cool")]
    
    
def test_clean_comment():
    test = "The quick 🏃 brown 🐴 fox 🦊 jumps over 😭 the lazy 💤😴 dog 🐶"
    assert clean_comment(test) == ['the', 'quick', 'brown', 'fox', 'jump', 'over', 'the', 'lazy', 'dog']