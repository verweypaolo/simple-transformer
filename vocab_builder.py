import re

from collections import Counter

text = """
The cat sat on the mat.
The cat ate the fish.
The fish was fresh.
"""

def regex_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def build_vocab(tokens, min_freq=1):
    counter = Counter(tokens) # dict of word counts, word: freq

    vocab_words = [word for word, freq in counter.items() if freq >= min_freq] # filter on freq
    special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'] # add special tokens

    vocab = special_tokens + sorted(vocab_words)

    stoi = {word: i for i, word in enumerate(vocab)} # map to int and back
    itos = {i: word for word, i in stoi.items()}

    return vocab, stoi, itos

def encode(tokens, stoi):
    '''returns integer for each token in tokens based on stoi map, returns integers for <UNK> if unknown'''
    return [stoi.get(token, stoi['<UNK>']) for token in tokens]

def decode(indices, itos):
    '''returns word for each integer in indices'''
    return [itos[i] for i in indices]


tokens = regex_tokenizer(text)
vocab, stoi, itos = build_vocab(tokens)
encoded = encode(tokens, stoi)
print(encoded)
decoded = decode(encoded, itos)
print(decoded)

# next sentence encoding
# Then continue with chatgpt 