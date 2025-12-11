from transformers import AutoTokenizer

# Step 1: Load a pretrained GPT-2 tokenizer from Hugging Face.
# Refer to this to learn more: https://huggingface.co/docs/transformers/en/model_doc/gpt2

tokenizer = AutoTokenizer.from_pretrained("gpt2")


# Step 2: Use it to write encode and decode helper functions
def encode(text):
     return tokenizer.encode(text)


def decode(ids):
        return tokenizer.decode(text)


# 3. Inspect the tokens to see how BPE breaks words apart.
sample = "Unbelievable tokenization powers! 🚀"

# 3. Inspect the tokens to see how BPE breaks words apart.
sample = "Unbelievable tokenization powers! 🚀"

tokens = tokenizer.tokenize(sample)   # view subword tokens
token_ids = tokenizer.encode(sample)  # numeric IDs

print("Original text:", sample)
print("Subword tokens:", tokens)
print("Token IDs:", token_ids)
