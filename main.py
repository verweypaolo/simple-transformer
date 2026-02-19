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

# 1. obtain tokens
tokens = regex_tokenizer(sample_text)

# 2. obtain vocabulary (all unique words in text, alphabetically sorted)
vocab = sorted(set(tokens))

# 3. Map each word to integer and back
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

# 4. Encode words
encoded = [stoi[word] for word in tokens]
print(encoded)


