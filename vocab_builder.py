import re
import math

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# let's create embeddings so the models can use our text
vocab_size = len(stoi)
embedding_dim = 32

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()

        # token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # positional embeddings
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

        # this code instantiates two Embedding objects with size input to be filled in forward function below.

    def forward(self, x):
        '''
        x_shape: (batch_size, context_length)
        '''

        batch_size, seq_length = x.shape

        # create position indices
        positions = torch.arange(seq_length, device=x.device) # this sets the tensor on the same device as my input tensor, [0, 1, 2, 3]
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(positions)
        #positions is a matrix of repeating rows, and position_emb is a 3 dimensional matrix of repeating two dimensional matrices, 
        # where the row of each two dimensional matrix is the position vector of 0 then 1 then 2 then 3

        return token_emb + position_emb
        # finally both embeddings are added to create a single embedding that stores both token and position information
    
embedding_layer = EmbeddingLayer(vocab_size, embedding_dim, context_length)
sample_input = torch.tensor(inputs[:2])
output = embedding_layer(sample_input) # calling model directly calls forward method in pytorch


# attention layer

# understand attention better
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key   = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embedding_dim)
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / math.sqrt(C)
        # shape: (B, T, T)

        # Create causal mask (prevent looking forward)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = attention @ V
        # shape: (B, T, C)

        return output

class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        
        self.attention = SelfAttention(embedding_dim)
        self.feedforward = FeedForward(embedding_dim)
        
        # also some learning applied here, hence why there are two separate layernorm functions
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Attention + residual
        x = x + self.attention(self.ln1(x)) # add to itself to make sure its refinement, not full replacement
        
        # Feedforward + residual
        x = x + self.feedforward(self.ln2(x))
        
        return x
    
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_length):
        super().__init__()

        self.embedding = EmbeddingLayer(vocab_size, embedding_dim, context_length)
        self.block = TransformerBlock(embedding_dim)
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.block(x)
        x = self.ln_final(x)
        logits = self.head(x)
        return logits


# now lets get to training!!

# turn our train/test samples into tensors
X = torch.tensor(inputs, dtype=torch.long)
Y = torch.tensor(targets, dtype=torch.long)

# move the tensors to the gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = X.to(device)
Y = Y.to(device)

# instantiate model and move to gpu
model = TinyGPT(vocab_size, embedding_dim=32, context_length=context_length)
model = model.to(device)

# train!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 500

for epoch in range(epochs):
    model.train() # tells pytorch we are training, important for things called Dropout and Batchnorm (?)
    
    # forward pass
    logits = model(X)

    # flattening required for cross_entropy calculation
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        Y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

