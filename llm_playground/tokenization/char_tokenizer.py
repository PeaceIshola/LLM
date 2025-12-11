import string

# Step 1: Create a vocabulary that includes all uppercase and lowercase letters.
vocab = []
char2id = {}
id2char = {}


# 1. Add two special tokens for unknown characters and padding
vocab.append("<PAD>")     # Used for padding if needed
vocab.append("<UNK>")     # Represents any non-supported character

# 2. Add lowercase a–z
for ch in string.ascii_lowercase:
    vocab.append(ch)

# 3. Add uppercase A–Z
for ch in string.ascii_uppercase:
    vocab.append(ch)

# 4. Build both mapping dictionaries
for idx, ch in enumerate(vocab):
    char2id[ch] = idx
    id2char[idx] = ch


print(f"Vocabulary size: {len(vocab)} (52 letters + 2 specials)")


# Step 2: Implement encode() and decode() functions to convert between text and IDs.
def encode(text):
    # convert text to list of IDs
    encoded_ids = []
    for ch in text:
        if ch in char2id:
            encoded_ids.append(char2id[ch])
        else:
            encoded_ids.append(char2id["<UNK>"])
    return encoded_ids


def decode(ids):
    # Convert list of IDs back to text
    decoded_chars = []
    for idx in ids:
        if idx in id2char:
            decoded_chars.append(id2char[idx])
        else:
            decoded_chars.append("<UNK>")
    return "".join(decoded_chars)


# Step 3: Test your tokenizer on a short sample word.
sample_text = "HelloWorld?!"       # includes an unsupported "!" to test <UNK>
encoded_output = encode(sample_text)
decoded_output = decode(encoded_output)

print("Original Text:", sample_text)
print("Encoded IDs  :", encoded_output)
print("Decoded Text :", decoded_output)
