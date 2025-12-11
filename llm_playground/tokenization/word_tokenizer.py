# Creating a tiny corpus. In practice, a corpus is generally the entire internet-scale dataset used for training.
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "Tokenization converts text to numbers",
    "Large language models predict the next token"
]

# Step 1: Build vocabulary (all unique words in the corpus) and mappings
vocab = []
word2id = {}
id2word = {}

for sentence in corpus:
  words = sentence.split()
  for word in words:
    if word not in vocab:
      vocab.append(word)

for idx, word in enumerate(vocab):
  word2id[word] = idx
  id2word[idx] = word

print(f"Vocabulary size: {len(vocab)} words")
print("First 15 vocab entries:", vocab[:15])


def encode(text):
    # converts text to token IDs

    words = text.split()
    token_ids = []

    for word in words:
        if word in word2id:
            token_ids.append(word2id[word])
        else:
            token_ids.append(-1)
    return token_ids


def decode(ids):
    # converts token IDs back to text

    words = []

    for idx in ids:
        if idx in id2word:                   # if ID exists in dictionary
            words.append(id2word[idx])       # append corresponding word
        else:
            words.append("<UNK>")            # unknown ID → placeholder token
    return " ".join(words)


# Step 3: Test your tokenizer with random sentences.
# Try a sentence with unseen words and see what happens (and how to fix it)

test_sentence_1 = "The quick brown fox"
encoded_1 = encode(test_sentence_1)
decoded_1 = decode(encoded_1)

print(encoded_1)
print(decoded_1)
