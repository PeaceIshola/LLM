import tiktoken

# Compare GPT-2 and GPT-4 tokenizers using tiktoken.

# Step 1: Load two tokenizers
gpt2_tok = tiktoken.get_encoding("gpt2")
gpt4_tok = tiktoken.get_encoding("cl100k_base")

# Step 2: Encode the same sentence with both and observe how they differ
sentence = "The 🌟 star-programmer implemented AGI overnight and i love it."

gpt2_ids = gpt2_tok.encode(sentence)
gpt4_ids = gpt4_tok.encode(sentence)

print("Original text:", sentence)

print("\n=== GPT-2 TOKENIZER (gpt2) ===")
print("Token IDs:", gpt2_ids)
print("Number of tokens:", len(gpt2_ids))

print("\n=== GPT-4 TOKENIZER (cl100k_base) ===")
print("Token IDs:", gpt4_ids)
print("Number of tokens:", len(gpt4_ids))
