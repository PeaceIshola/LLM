import torch
from transformers import GPT2LMHeadModel

# Step 1: load the smallest GPT-2 model (124M parameters) using the Hugging Face transformers library.
# Refer to: https://huggingface.co/docs/transformers/en/model_doc/gpt2
# Step 1: load the smallest GPT-2 model (124M parameters)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: inspect (print) the first Transformer block
print(model.transformer.h[0])


# Step 1: Create a small dummy input with a sequence of 8 random token IDs.
batch_size = 2
seq_len = 16
dummy_ids = torch.randint(low=0, high=model.config.vocab_size, size=(batch_size, seq_len))

# Step 2: Convert token IDs into embeddings (token + positional)
token_embeds = model.transformer.wte(dummy_ids)                      # (B, T, hidden)
pos_ids = torch.arange(seq_len).unsqueeze(0)                         # (1, T)
pos_embeds = model.transformer.wpe(pos_ids)                          # (1, T, hidden)
hidden_states = token_embeds + pos_embeds                            # (B, T, hidden)

# Step 3: Pass the embeddings through a single Transformer block
block_out = model.transformer.h[0](hidden_states)[0]                 # first block output

# Step 4: Inspect the result
print("Output shape:", block_out.shape)


# Print the name of all layers inside gpt.transformer.
# You may find this helpful: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_children

# Print the name of all layers inside gpt.transformer.
for name, layer in model.transformer.named_children():
    print(name, ":", type(layer))
