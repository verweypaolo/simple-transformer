import re

from transformers import AutoTokenizer

sample_text = 'The cat sat on the mat. The cat ate the fish.'

def whitespace_tokenizer(text):
    text = text.lower()
    tokens = text.split()
    return tokens

def regex_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def hugging_face_tokenizer(text):
    '''returns integers'''
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text)
    return tokens

print(hugging_face_tokenizer(sample_text))

