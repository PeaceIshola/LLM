import torch, torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Step 1: Load GPT-2 model and its tokenizer

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


# Step 2: Tokenize input text
text = "Hello my name"

text = "Hello my name"
inputs = tokenizer(text, return_tensors="pt")   # contains 'input_ids' and 'attention_mask'


# Step 3: Pass the input IDs to the model

with torch.no_grad():
    outputs = model(**inputs)                   # forward pass
logits = outputs.logits                         # shape: (batch, seq_len, vocab_size)

print("Logits shape:", logits.shape)


# Step 4: Predict the next token
# We take the logits from the final position, apply softmax to get probabilities,
# and then extract the top 5 most likely next tokens. You may find F.softmax and torch.topk helpful in your implementation.

last_token_logits = logits[0, -1, :]            # logits for final token position
probs = F.softmax(last_token_logits, dim=-1)    # convert to probabilities
top_probs, top_ids = torch.topk(probs, k=5)     # top 5 token IDs

print("\nTop 5 predicted next tokens:")
for prob, token_id in zip(top_probs, top_ids):
    print(f"{tokenizer.decode([token_id])!r}  ->  probability: {prob.item():.4f}")
