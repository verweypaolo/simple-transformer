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

def add_bos_eos_and_tokenize(text):
    sentences = text.strip().split('\n')
    all_tokens = []

    for sentence in sentences:
        sentence_tokens = regex_tokenizer(sentence)
        all_tokens += ['<BOS>'] + sentence_tokens + ['<EOS>']
    
    return all_tokens

def build_vocab(tokens, min_freq=1):
    counter = Counter(tokens) # dict of word counts, word: freq

    vocab_words = [word for word, freq in counter.items() if freq >= min_freq] # filter on freq
    special_tokens = ['<PAD>', '<UNK>'] # add special tokens
    # need to remove these special tokens if they are included in the training text, otherwise stoi will be wrongly constructed

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


tokens = add_bos_eos_and_tokenize(text)

vocab, stoi, itos = build_vocab(tokens)

encoded = encode(tokens, stoi)
decoded = decode(encoded, itos)

# let's create training sequences
# these are sequences of length context_length that include as target the next token in the training data 'text'

def create_sequences(encoded_tokens, context_length):
    inputs = []
    targets = []

    for i in range(len(encoded_tokens) - context_length):
        input_seq = encoded_tokens[i:i+context_length]
        target_seq = encoded_tokens[i+1:i+context_length+1]
        
        inputs.append(input_seq)
        targets.append(target_seq)

    return inputs, targets
# returning two lists of context_length sized sequences (also lists)

context_length = 4
inputs, targets = create_sequences(encoded, context_length)

print(inputs[0], targets[0])